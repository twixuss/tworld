#define _CRT_SECURE_NO_WARNINGS
#define TL_IMPL
#define TL_DEBUG 1
#include <tl/common.h>
#include <tl/console.h>
#include <tl/file.h>
#include <tl/input.h>
#include <tl/vector.h>
#include <tl/win32.h>
#include <tl/time.h>
#include <tl/math_random.h>
#include <tl/hash_map.h>
#include <tl/hash_set.h>
#include <tl/profiler.h>
#include <tl/cpu.h>
#include <tl/simd.h>

using namespace tl;

#include <dxgi1_6.h>
#include <d3dcompiler.h>
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3dcompiler.lib")

#include "cbuffer.h"
#include "input.h"
#include "common.h"

HWND hwnd;
IDXGISwapChain *swap_chain = 0;
ID3D11Device *device = 0;
ID3D11DeviceContext *immediate_context = 0;

ID3D11RenderTargetView *back_buffer = 0;
ID3D11DepthStencilView *depth_stencil = 0;

ID3D11RenderTargetView   *sky_rt  = 0;
ID3D11ShaderResourceView *sky_srv = 0;

ID3D11RenderTargetView   *shadow_rtv = 0;
ID3D11DepthStencilView   *shadow_dsv = 0;
ID3D11ShaderResourceView *shadow_srv = 0;
u32 const shadow_map_size = 1024;
u32 const shadow_world_size = 256;

ID3D11InfoQueue* debug_info_queue = 0;

ID3D11RasterizerState *wireframe_rasterizer;
bool wireframe_rasterizer_enabled;

ID3D11BlendState *alpha_blend;

ID3D11VertexShader *chunk_vs = 0;
ID3D11PixelShader *chunk_solid_ps = 0;
ID3D11PixelShader *chunk_wire_ps = 0;

ID3D11VertexShader *cursor_vs = 0;
ID3D11PixelShader  *cursor_ps = 0;

ID3D11VertexShader *sky_vs = 0;
ID3D11PixelShader  *sky_ps = 0;

ID3D11VertexShader *blit_vs = 0;
ID3D11PixelShader  *blit_ps = 0;

ID3D11VertexShader *shadow_vs = 0;
ID3D11PixelShader  *shadow_ps = 0;

struct alignas(16) FrameCbuffer {
    m4 mvp;
    m4 rotproj;
	m4 light_vp_matrix;
	v3f campos;
	f32 _;
	v3f ldir;
};
#define FRAME_CBUFFER_STR R"(
cbuffer _ : register(b0) {
    float4x4 c_mvp;
    float4x4 c_rotproj;
	float4x4 light_vp_matrix;
	float3 c_campos;
	float _;
	float3 c_ldir;
}
)"

CBuffer<FrameCbuffer> frame_cbuffer;

struct alignas(16) ChunkCbuffer {
    v3f relative_position;
	f32 was_remeshed;
    v3f actual_position;
};

CBuffer<ChunkCbuffer> chunk_cbuffer;


Profiler profiler;
bool profile_frame;


ThreadPool thread_pool;

namespace neighbor {
u32 const x = 0;
u32 const y = 1;
u32 const z = 2;
u32 const yz = 3;
u32 const xz = 4;
u32 const xy = 5;
u32 const xyz = 6;
}

struct ChunkRelativePosition {
	v3s chunk;
	v3f local;

	ChunkRelativePosition operator-() const {
		return {
			-chunk,
			-local,
		};
	}
	ChunkRelativePosition &operator+=(v3f that) {
		local += that;
		auto chunk_offset = floor(floor_to_int(local), CHUNKW);
		chunk += chunk_offset / CHUNKW;
		local -= (v3f)chunk_offset;
		return *this;
	}
	ChunkRelativePosition &operator+=(ChunkRelativePosition that) {
		local += that.local;
		auto chunk_offset = floor(floor_to_int(local), CHUNKW);
		chunk += that.chunk + chunk_offset / CHUNKW;
		local -= (v3f)chunk_offset;
		return *this;
	}

	ChunkRelativePosition operator+(v3f that) const { return ChunkRelativePosition(*this) += that; }
	ChunkRelativePosition operator-(v3f that) const { return ChunkRelativePosition(*this) -= that; }
	ChunkRelativePosition &operator-=(v3f that) { return *this += -that; }

	ChunkRelativePosition operator+(ChunkRelativePosition that) const { return ChunkRelativePosition(*this) += that; }
	ChunkRelativePosition operator-(ChunkRelativePosition that) const { return ChunkRelativePosition(*this) -= that; }
	ChunkRelativePosition &operator-=(ChunkRelativePosition that) { return *this += -that; }

	v3f to_v3f() {
		return (v3f)(chunk * CHUNKW) + local;
	}
};

ChunkRelativePosition camera_position = {0,1,0};
ChunkRelativePosition prev_camera_position;

enum class CameraMode {
	walk,
	fly,
};

CameraMode camera_mode;

v3f target_camera_angles;
v3f camera_angles;

f32 camera_fov = 90;

struct NeighborMask {
	bool x : 1;
	bool y : 1;
	bool z : 1;
	auto operator<=>(NeighborMask const &) const = default;
};

struct Chunk {
	ID3D11ShaderResourceView *vertex_buffer = 0;
	u32 vert_count = 0;
	u32 lod_width = 1;
	Mutex mutex;
	bool sdf_generated = false;
	List<v3f> vertices;
	NeighborMask neighbor_mask = {};
	f32 time_since_remesh = 0;
#if 0
	Array<Array<Array<s8, CHUNKW>, CHUNKW>, CHUNKW> sdf;
#else
	s8 sdf0[CHUNKW][CHUNKW][CHUNKW];
	s8 sdf1[CHUNKW/2][CHUNKW/2][CHUNKW/2];
	s8 sdf2[CHUNKW/4][CHUNKW/4][CHUNKW/4];
	s8 sdf3[CHUNKW/8][CHUNKW/8][CHUNKW/8];
	s8 sdf4[CHUNKW/16][CHUNKW/16][CHUNKW/16];
	s8 sdf5[CHUNKW/32][CHUNKW/32][CHUNKW/32];
#endif

	Chunk() = default;
	Chunk(Chunk const &) = delete;
	Chunk(Chunk &&) = delete;
};

Chunk (*_chunks)[DRAWD*2+1][DRAWD*2+1][DRAWD*2+1];

#define chunks (*_chunks)

int _____ = sizeof Chunk;

Chunk &get_chunk(s32 x, s32 y, s32 z) {
	s32 const s = DRAWD*2+1;
	x += DRAWD;
	y += DRAWD;
	z += DRAWD;
	return chunks[frac(x, s)][frac(y, s)][frac(z, s)];
}
Chunk &get_chunk(v3s v) {
	return get_chunk(v.x, v.y, v.z);
}

v3s get_chunk_position(Chunk *chunk) {
	auto index = chunk - (Chunk *)&chunks[0][0][0];
	s32 const s = DRAWD*2+1;

	v3s v = {
		(index / (s * s)) % s,
		(index / s) % s,
		index % s,
	};
	v -= DRAWD;

	while (v.x < -DRAWD) v.x += s;
	while (v.y < -DRAWD) v.y += s;
	while (v.z < -DRAWD) v.z += s;
	while (v.x > DRAWD) v.x -= s;
	while (v.y > DRAWD) v.y -= s;
	while (v.z > DRAWD) v.z -= s;

	return v;
}

template <class Fn>
void for_each_chunk(Fn &&fn) {
	timed_function(profiler, profile_frame);

	for (s32 x = camera_position.chunk.x-DRAWD; x <= camera_position.chunk.x+DRAWD; ++x) {
	for (s32 y = camera_position.chunk.y-DRAWD; y <= camera_position.chunk.y+DRAWD; ++y) {
	for (s32 z = camera_position.chunk.z-DRAWD; z <= camera_position.chunk.z+DRAWD; ++z) {
		fn(get_chunk(x,y,z), {x,y,z});
	}
	}
	}
}

NeighborMask get_neighbor_mask(v3s position) {
	auto _100 = get_chunk(position + v3s{1,0,0}).sdf_generated;
	auto _010 = get_chunk(position + v3s{0,1,0}).sdf_generated;
	auto _001 = get_chunk(position + v3s{0,0,1}).sdf_generated;
	auto _011 = get_chunk(position + v3s{0,1,1}).sdf_generated;
	auto _101 = get_chunk(position + v3s{1,0,1}).sdf_generated;
	auto _110 = get_chunk(position + v3s{1,1,0}).sdf_generated;
	auto _111 = get_chunk(position + v3s{1,1,1}).sdf_generated;

	NeighborMask mask;
	mask.x =
		(position - camera_position.chunk).x != DRAWD &&
		_100 &&
		_110 &&
		_101 &&
		_111;
	mask.y =
		(position - camera_position.chunk).y != DRAWD &&
		_010 &&
		_110 &&
		_011 &&
		_111;
	mask.z =
		(position - camera_position.chunk).z != DRAWD &&
		_001 &&
		_101 &&
		_011 &&
		_111;
	return mask;
}

struct Vertex {
	v3f position;
	v3f normal;
	u32 parent_vertex;
};

#if 0

	f32 sdf[lodw+2][lodw+2][lodw+2]{};
#if 0
	if constexpr (true) {//(false && lodw == 16) {
		timed_block(profiler, profile_frame, "sdf generation");
		for (s32 x = 0; x < lodw+2; ++x) {
		for (s32 y = 0; y < lodw+2; ++y) {
		for (s32 z = 0; z < lodw+2; ++z) {
			f32 d = 0;

			v3s gg = v3s{x,y,z}*CHUNKW/lodw + chunk.position * CHUNKW;


			for (s32 ox = 0; ox < (1<<(log2(CHUNKW)-log2(lodw))); ++ox) {
			for (s32 oy = 0; oy < (1<<(log2(CHUNKW)-log2(lodw))); ++oy) {
			for (s32 oz = 0; oz < (1<<(log2(CHUNKW)-log2(lodw))); ++oz) {
				v3s g = gg + v3s{ox,oy,oz};

				s32 scale = 1;
				for (s32 i = 0; i < 4; ++i) {
					d += (value_noise_v3s_smooth({
						(s32)(g.x + 0xa133b78c),
						(s32)(g.y + 0x2d462e4f),
						(s32)(g.z + 0x9f83e86d)
					}, CHUNKW*scale/4) - 0.5f + (CHUNKW/2 - g.y) * 0.00125f) * scale;
					scale *= 2;
				}
			}
			}
			}

			if (-0.01f <= d && d <= 0.01f) {
				d = 0.01f;
			}

			sdf[x][y][z] = d;
		}
		}
		}
	} // else
#else
	{
		timed_block(profiler, profile_frame, "sdf generation");
#if 0
		for (s32 x = 0; x < (lodw+2) * (1<<(log2(CHUNKW)-log2(lodw))); ++x) {
		for (s32 y = 0; y < (lodw+2) * (1<<(log2(CHUNKW)-log2(lodw))); ++y) {
		for (s32 z = 0; z < (lodw+2) * (1<<(log2(CHUNKW)-log2(lodw))); ++z) {
			f32 d = 0;

			v3s g = v3s{x,y,z} + chunk.position * CHUNKW;

			s32 scale = 1;
			for (s32 i = 0; i < 4; ++i) {
				d += (value_noise_v3s_smooth({
					(s32)(g.x + 0xa133b78c),
					(s32)(g.y + 0x2d462e4f),
					(s32)(g.z + 0x9f83e86d)
				}, CHUNKW*scale/4) - 0.5f + (CHUNKW/2 - g.y) * 0.00125f) * scale;
				scale *= 2;
			}

			if (-0.01f <= d && d <= 0.01f) {
				d = 0.01f;
			}

			sdf[x*lodw/CHUNKW][y*lodw/CHUNKW][z*lodw/CHUNKW] += d;
		}
		}
		}
#else
		for (s32 x = 0; x < lodw+2; ++x) {
		for (s32 y = 0; y < lodw+2; ++y) {
		for (s32 z = 0; z < lodw+2; ++z) {
			f32 d = 0;

			v3s g = v3s{x,y,z} + chunk.position * CHUNKW;

			s32 scale = 1;
			for (s32 i = 0; i < 4; ++i) {
				d += (value_noise_v3s_smooth({
					(s32)(g.x + 0xa133b78c),
					(s32)(g.y + 0x2d462e4f),
					(s32)(g.z + 0x9f83e86d)
				}, CHUNKW*scale/4) - 0.5f + (CHUNKW/2 - g.y) * 0.00125f) * scale;
				scale *= 2;
			}

			d *= 1000;

			if (-0.01f <= d && d <= 0.01f) {
				d = 0.01f;
			}

			sdf[x][y][z] += d;
		}
		}
		}
#endif
	}
#endif

#endif

u64 total_time_wasted_on_generating_sdfs;
u64 total_sdfs_generated;

s32x8 randomize(s32x8 v) {
	v = vec32_xor(v, s32x8_set1(0x55555555u));
	v = s32x8_mul(v, s32x8_set1(u32_random_primes[0]));
	v = vec32_xor(v, s32x8_set1(0x33333333u));
	v = s32x8_mul(v, s32x8_set1(u32_random_primes[1]));
	return v;
}

f32x8 random_f32x8(s32x8 x, s32x8 y, s32x8 z) {
	x = randomize(x);
	x = randomize(vec32_xor(x, y));
	x = randomize(vec32_xor(x, z));
	return f32x8_mul(s32x8_to_f32x8(s32x8_slri(x, 8)), f32x8_set1(1.0f / ((1 << 24) - 1)));
}

void filter_sdf(Chunk &chunk) {
	timed_function(profiler, profile_frame);

#define LOD(div, dst, src) \
	for (s32 sx = 0, dx = 0; sx < CHUNKW/div; sx += 2, dx += 1) { \
	for (s32 sy = 0, dy = 0; sy < CHUNKW/div; sy += 2, dy += 1) { \
	for (s32 sz = 0, dz = 0; sz < CHUNKW/div; sz += 2, dz += 1) { \
		chunk.dst[dx][dy][dz] = ( \
			chunk.src[sx+0][sy+0][sz+0] + \
			chunk.src[sx+0][sy+0][sz+1] + \
			chunk.src[sx+0][sy+1][sz+0] + \
			chunk.src[sx+0][sy+1][sz+1] + \
			chunk.src[sx+1][sy+0][sz+0] + \
			chunk.src[sx+1][sy+0][sz+1] + \
			chunk.src[sx+1][sy+1][sz+0] + \
			chunk.src[sx+1][sy+1][sz+1] \
			) >> 3; \
	} \
	} \
	}

	LOD(1,  sdf1, sdf0);
	LOD(2,  sdf2, sdf1);
	LOD(4,  sdf3, sdf2);
	LOD(8,  sdf4, sdf3);
	LOD(16, sdf5, sdf4);

#undef LOD
}

void generate_sdf(Chunk &chunk, v3s chunk_position) {
	timed_function(profiler, profile_frame);
	auto start_time = get_performance_counter();

	defer {
		chunk.sdf_generated = true;
		atomic_add(&total_time_wasted_on_generating_sdfs, get_performance_counter() - start_time);
		atomic_increment(&total_sdfs_generated);
	};

	if (chunk_position.y > 4) {
		memset(chunk.sdf0, 0, sizeof chunk.sdf0);
		memset(chunk.sdf1, 0, sizeof chunk.sdf1);
		memset(chunk.sdf2, 0, sizeof chunk.sdf2);
		memset(chunk.sdf3, 0, sizeof chunk.sdf3);
		memset(chunk.sdf4, 0, sizeof chunk.sdf4);
		memset(chunk.sdf5, 0, sizeof chunk.sdf5);
		return;
	}

	if (chunk_position.y < -4) {
		memset(chunk.sdf0, 255, sizeof chunk.sdf0);
		memset(chunk.sdf1, 255, sizeof chunk.sdf1);
		memset(chunk.sdf2, 255, sizeof chunk.sdf2);
		memset(chunk.sdf3, 255, sizeof chunk.sdf3);
		memset(chunk.sdf4, 255, sizeof chunk.sdf4);
		memset(chunk.sdf5, 255, sizeof chunk.sdf5);
		return;
	}

	f32 sdf_max_abs = -1;
	f32 sdf_max     = min_value<f32>;
	f32 tmp[CHUNKW][CHUNKW][CHUNKW];

	for (s32 x = 0; x < CHUNKW; ++x) {
	for (s32 y = 0; y < CHUNKW; ++y) {
#if 1
	for (s32 z = 0; z < CHUNKW; z += 8) {
		f32x8 d = {};
		s32 scale = 1;
		for (s32 i = 0; i < 8; ++i) {
			s32x8 gx = s32x8_set1(x + chunk_position.x * CHUNKW);
			s32x8 gy = s32x8_set1(y + chunk_position.y * CHUNKW);
			s32x8 gz = s32x8_add(s32x8_set1(z + chunk_position.z * CHUNKW), s32x8_set(0,1,2,3,4,5,6,7));

			s32 step = scale*2;

			floor(1, 2);

			s32x8 fx = s32x8_floor(gx, s32x8_set1(step));
			s32x8 fy = s32x8_floor(gy, s32x8_set1(step));
			s32x8 fz = s32x8_floor(gz, s32x8_set1(step));
			s32x8 ix = s32x8_div(fx, s32x8_set1(step));
			s32x8 iy = s32x8_div(fy, s32x8_set1(step));
			s32x8 iz = s32x8_div(fz, s32x8_set1(step));

			f32x8 lx = f32x8_mul(s32x8_to_f32x8(s32x8_sub(gx, fx)), f32x8_set1(1.0f / step));
			f32x8 ly = f32x8_mul(s32x8_to_f32x8(s32x8_sub(gy, fy)), f32x8_set1(1.0f / step));
			f32x8 lz = f32x8_mul(s32x8_to_f32x8(s32x8_sub(gz, fz)), f32x8_set1(1.0f / step));

			// x*x*x*(x*(6*x - 15) + 10);

			f32x8 tx = f32x8_mul(lx, f32x8_mul(lx, f32x8_mul(lx, f32x8_muladd(lx, f32x8_muladd(lx, f32x8_set1(6), f32x8_set1(-15)), f32x8_set1(10)))));
			f32x8 ty = f32x8_mul(ly, f32x8_mul(ly, f32x8_mul(ly, f32x8_muladd(ly, f32x8_muladd(ly, f32x8_set1(6), f32x8_set1(-15)), f32x8_set1(10)))));
			f32x8 tz = f32x8_mul(lz, f32x8_mul(lz, f32x8_mul(lz, f32x8_muladd(lz, f32x8_muladd(lz, f32x8_set1(6), f32x8_set1(-15)), f32x8_set1(10)))));

			f32x8 left_bottom_back   = random_f32x8(s32x8_add(ix, s32x8_set1(0)), s32x8_add(iy, s32x8_set1(0)), s32x8_add(iz, s32x8_set1(0)));
			f32x8 right_bottom_back  = random_f32x8(s32x8_add(ix, s32x8_set1(1)), s32x8_add(iy, s32x8_set1(0)), s32x8_add(iz, s32x8_set1(0)));
			f32x8 left_top_back      = random_f32x8(s32x8_add(ix, s32x8_set1(0)), s32x8_add(iy, s32x8_set1(1)), s32x8_add(iz, s32x8_set1(0)));
			f32x8 right_top_back     = random_f32x8(s32x8_add(ix, s32x8_set1(1)), s32x8_add(iy, s32x8_set1(1)), s32x8_add(iz, s32x8_set1(0)));
			f32x8 left_bottom_front  = random_f32x8(s32x8_add(ix, s32x8_set1(0)), s32x8_add(iy, s32x8_set1(0)), s32x8_add(iz, s32x8_set1(1)));
			f32x8 right_bottom_front = random_f32x8(s32x8_add(ix, s32x8_set1(1)), s32x8_add(iy, s32x8_set1(0)), s32x8_add(iz, s32x8_set1(1)));
			f32x8 left_top_front     = random_f32x8(s32x8_add(ix, s32x8_set1(0)), s32x8_add(iy, s32x8_set1(1)), s32x8_add(iz, s32x8_set1(1)));
			f32x8 right_top_front    = random_f32x8(s32x8_add(ix, s32x8_set1(1)), s32x8_add(iy, s32x8_set1(1)), s32x8_add(iz, s32x8_set1(1)));

			f32x8 left_bottom  = f32x8_lerp(left_bottom_back,  left_bottom_front,  tz);
			f32x8 right_bottom = f32x8_lerp(right_bottom_back, right_bottom_front, tz);
			f32x8 left_top     = f32x8_lerp(left_top_back,     left_top_front,     tz);
			f32x8 right_top    = f32x8_lerp(right_top_back,    right_top_front,    tz);

			f32x8 left  = f32x8_lerp(left_bottom,  left_top,  ty);
			f32x8 right = f32x8_lerp(right_bottom, right_top, ty);

			f32x8 h = f32x8_lerp(left, right, tx);
			h = f32x8_mul(f32x8_sub(h, f32x8_add(f32x8_set1(0.5f), f32x8_mul(s32x8_to_f32x8(gy), f32x8_set1(1)))), f32x8_set1(scale));
			//h = f32x8_mul(f32x8_sub(h, f32x8_mul(s32x8_to_f32x8(gy), f32x8_set1(8))), f32x8_set1(scale));
			//d = f32x8_add(d, f32x8_add(f32x8_set1(-0.5f), f32x8_mul(s32x8_to_f32x8(s32x8_sub(s32x8_set1(CHUNKW/2), gy)), f32x8_set1(0.0125f))));
			d = f32x8_add(d, h);

			scale *= 2;
		}

		f32x8_store(&tmp[x][y][z], d);
#else
	for (s32 z = 0; z < CHUNKW; ++z) {
		v3s g = v3s{x,y,z} + chunk_position * CHUNKW;

		f32 d = 0;
#if 0
		d = (value_noise_v3s_smooth({
			(s32)(g.x),
			(s32)(g.y),
			(s32)(g.z)
		}, 16) - 0.5f + (CHUNKW/2 - g.y) * 0.0125f);

		//d = 31.25-(x+y+z);
#else
		s32 scale = 1;
		for (s32 i = 0; i < 4; ++i) {
			d += (value_noise_v3s_smooth({
				(s32)(g.x + 0xa133b78c),
				(s32)(g.y + 0x2d462e4f),
				(s32)(g.z + 0x9f83e86d)
			}, CHUNKW*scale/16) - 0.5f + (CHUNKW/2 - g.y) * 0.0125f) * scale;
			scale *= 2;
		}
#endif

		//if (-0.01f <= d && d <= 0.01f) {
		//	d = 0.01f;
		//}

		s8 s = (s8)map_clamped(d, -1.f, 1.f, (f32)min_value<s8>, (f32)max_value<s8>);
		if (s == 0)
			s = 1;
		chunk.sdf[x][y][z] = s;
#endif
	}
	}
	}

	constexpr u8 edges[][2] {
		{0b000, 0b001},
		{0b010, 0b011},
		{0b100, 0b101},
		{0b110, 0b111},
		{0b000, 0b010},
		{0b001, 0b011},
		{0b100, 0b110},
		{0b101, 0b111},
		{0b000, 0b100},
		{0b001, 0b101},
		{0b010, 0b110},
		{0b011, 0b111},
	};

	{
		for (s32 x = 0; x < CHUNKW-1; ++x) {
		for (s32 y = 0; y < CHUNKW-1; ++y) {
		for (s32 z = 0; z < CHUNKW-1; ++z) {
			f32 d[8];
			d[0b000] = tmp[x+0][y+0][z+0];
			d[0b001] = tmp[x+0][y+0][z+1];
			d[0b010] = tmp[x+0][y+1][z+0];
			d[0b011] = tmp[x+0][y+1][z+1];
			d[0b100] = tmp[x+1][y+0][z+0];
			d[0b101] = tmp[x+1][y+0][z+1];
			d[0b110] = tmp[x+1][y+1][z+0];
			d[0b111] = tmp[x+1][y+1][z+1];
			u8 e =
				(u8)(d[0b000] > 0) +
				(u8)(d[0b001] > 0) +
				(u8)(d[0b010] > 0) +
				(u8)(d[0b011] > 0) +
				(u8)(d[0b100] > 0) +
				(u8)(d[0b101] > 0) +
				(u8)(d[0b110] > 0) +
				(u8)(d[0b111] > 0);


			if (e != 0 && e != 8) {
				for (auto &edge : edges) {
					auto a = edge[0];
					auto b = edge[1];
					if ((d[a] > 0) != (d[b] > 0)) {
						sdf_max_abs = max(sdf_max_abs, absolute(d[a]), absolute(d[b]));
						sdf_max     = max(sdf_max_abs, d[a], d[b]);
					}
				}
			}
		}
		}
		}
	}

	if (sdf_max_abs == -1) {
		sdf_max_abs = 1;
	}
	for (s32 x = 0; x < CHUNKW; ++x) {
	for (s32 y = 0; y < CHUNKW; ++y) {
	for (s32 z = 0; z < CHUNKW; ++z) {
		chunk.sdf0[x][y][z] = (s8)map_clamped(tmp[x][y][z], -sdf_max_abs, sdf_max_abs, -128.f, 127.f);
	}
	}
	}

	filter_sdf(chunk);
}

v3f sdf_gradient(s8 (&sdf)[8], v3f point) {
	v3f p00 = {sdf[0b001], sdf[0b010], sdf[0b100]};
    v3f n00 = {sdf[0b000], sdf[0b000], sdf[0b000]};

    v3f p10 = {sdf[0b101], sdf[0b011], sdf[0b110]};
    v3f n10 = {sdf[0b100], sdf[0b001], sdf[0b010]};

    v3f p01 = {sdf[0b011], sdf[0b110], sdf[0b101]};
    v3f n01 = {sdf[0b010], sdf[0b100], sdf[0b001]};

    v3f p11 = {sdf[0b111], sdf[0b111], sdf[0b111]};
    v3f n11 = {sdf[0b110], sdf[0b101], sdf[0b011]};

    // Each dimension encodes an edge delta, giving 12 in total.
    auto d00 = p00 - n00; // Edges (0b00x, 0b0y0, 0bz00)
    auto d10 = p10 - n10; // Edges (0b10x, 0b0y1, 0bz10)
    auto d01 = p01 - n01; // Edges (0b01x, 0b1y0, 0bz01)
    auto d11 = p11 - n11; // Edges (0b11x, 0b1y1, 0bz11)

	auto neg = v3f{1,1,1} - point;

    // Do bilinear interpolation between 4 edges in each dimension.
    return neg.yzx() * neg.zxy() * d00
        + neg.yzx() * point.zxy() * d10
        + point.yzx() * neg.zxy() * d01
        + point.yzx() * point.zxy() * d11;
}

template <>
u64 get_hash(Chunk *const &chunk) {
	return chunk - &chunks[0][0][0];
}

LinearSet<Chunk *> chunks_with_meshes;
Mutex chunks_with_meshes_mutex;

template <u32 lodw>
void generate_chunk_lod(Chunk &chunk, v3s chunk_position) {
	v3s lbounds = V3s(lodw);
	if constexpr (lodw <= 2) {
		if (chunk.neighbor_mask.x) lbounds.x += 2;
		if (chunk.neighbor_mask.y) lbounds.y += 2;
		if (chunk.neighbor_mask.z) lbounds.z += 2;
	} else {
		if (chunk.neighbor_mask.x) lbounds.x += 4;
		if (chunk.neighbor_mask.y) lbounds.y += 4;
		if (chunk.neighbor_mask.z) lbounds.z += 4;
	}

	auto get_sdf = [&] (Chunk *chunk) {
		     if constexpr (lodw == 32) return chunk->sdf0;
		else if constexpr (lodw == 16) return chunk->sdf1;
		else if constexpr (lodw ==  8) return chunk->sdf2;
		else if constexpr (lodw ==  4) return chunk->sdf3;
		else if constexpr (lodw ==  2) return chunk->sdf4;
		else if constexpr (lodw ==  1) return chunk->sdf5;
		else static_assert(false);
	};

	auto sdf = [&](s32 x, s32 y, s32 z) {
		x -= (x == lodw*2);
		y -= (y == lodw*2);
		z -= (z == lodw*2);
		if (x < lodw) {
			if (y < lodw) {
				if (z < lodw) {
					return get_sdf(&chunk)[x][y][z];
				} else {
					return get_sdf(&get_chunk(chunk_position+v3s{0,0,1}))[x][y][z-lodw];
				}
			} else {
				if (z < lodw) {
					return get_sdf(&get_chunk(chunk_position+v3s{0,1,0}))[x][y-lodw][z];
				} else {
					return get_sdf(&get_chunk(chunk_position+v3s{0,1,1}))[x][y-lodw][z-lodw];
				}
			}
		} else {
			if (y < lodw) {
				if (z < lodw) {
					return get_sdf(&get_chunk(chunk_position+v3s{1,0,0}))[x-lodw][y][z];
				} else {
					return get_sdf(&get_chunk(chunk_position+v3s{1,0,1}))[x-lodw][y][z-lodw];
				}
			} else {
				if (z < lodw) {
					return get_sdf(&get_chunk(chunk_position+v3s{1,1,0}))[x-lodw][y-lodw][z];
				} else {
					return get_sdf(&get_chunk(chunk_position+v3s{1,1,1}))[x-lodw][y-lodw][z-lodw];
				}
			}
		}
	};

#if 0
	s32 sdf_[lodw+4][lodw+4][lodw+4]{};
	{
		timed_block(profiler, profile_frame, "sdf filtering");
		for (s32 x = 0; x < CHUNKW; ++x) {
		for (s32 y = 0; y < CHUNKW; ++y) {
		for (s32 z = 0; z < CHUNKW; ++z) {
			sdf_[x*lodw/CHUNKW][y*lodw/CHUNKW][z*lodw/CHUNKW] += chunk.sdf0[x][y][z];
		}
		}
		}
		Chunk *neighbor;

		neighbor = &get_chunk(chunk_position+v3s{1,0,0});
		for (s32 x = CHUNKW; x < lbounds.x*CHUNKW/lodw; ++x) {
		for (s32 y = 0; y < CHUNKW; ++y) {
		for (s32 z = 0; z < CHUNKW; ++z) {
			sdf_[x*lodw/CHUNKW][y*lodw/CHUNKW][z*lodw/CHUNKW] += neighbor->sdf0[min(CHUNKW-1,x-CHUNKW)][y][z];
		}
		}
		}
		neighbor = &get_chunk(chunk_position+v3s{0,1,0});
		for (s32 x = 0; x < CHUNKW; ++x) {
		for (s32 y = CHUNKW; y < lbounds.y*CHUNKW/lodw; ++y) {
		for (s32 z = 0; z < CHUNKW; ++z) {
			sdf_[x*lodw/CHUNKW][y*lodw/CHUNKW][z*lodw/CHUNKW] += neighbor->sdf0[x][min(CHUNKW-1,y-CHUNKW)][z];
		}
		}
		}
		neighbor = &get_chunk(chunk_position+v3s{1,1,0});
		for (s32 x = CHUNKW; x < lbounds.x*CHUNKW/lodw; ++x) {
		for (s32 y = CHUNKW; y < lbounds.y*CHUNKW/lodw; ++y) {
		for (s32 z = 0; z < CHUNKW; ++z) {
			sdf_[x*lodw/CHUNKW][y*lodw/CHUNKW][z*lodw/CHUNKW] += neighbor->sdf0[min(CHUNKW-1,x-CHUNKW)][min(CHUNKW-1,y-CHUNKW)][z];
		}
		}
		}
		neighbor = &get_chunk(chunk_position+v3s{0,0,1});
		for (s32 x = 0; x < CHUNKW; ++x) {
		for (s32 y = 0; y < CHUNKW; ++y) {
		for (s32 z = CHUNKW; z < lbounds.z*CHUNKW/lodw; ++z) {
			sdf_[x*lodw/CHUNKW][y*lodw/CHUNKW][z*lodw/CHUNKW] += neighbor->sdf0[x][y][min(CHUNKW-1,z-CHUNKW)];
		}
		}
		}
		neighbor = &get_chunk(chunk_position+v3s{1,0,1});
		for (s32 x = CHUNKW; x < lbounds.x*CHUNKW/lodw; ++x) {
		for (s32 y = 0; y < CHUNKW; ++y) {
		for (s32 z = CHUNKW; z < lbounds.z*CHUNKW/lodw; ++z) {
			sdf_[x*lodw/CHUNKW][y*lodw/CHUNKW][z*lodw/CHUNKW] += neighbor->sdf0[min(CHUNKW-1,x-CHUNKW)][y][min(CHUNKW-1,z-CHUNKW)];
		}
		}
		}
		neighbor = &get_chunk(chunk_position+v3s{0,1,1});
		for (s32 x = 0; x < CHUNKW; ++x) {
		for (s32 y = CHUNKW; y < lbounds.y*CHUNKW/lodw; ++y) {
		for (s32 z = CHUNKW; z < lbounds.z*CHUNKW/lodw; ++z) {
			sdf_[x*lodw/CHUNKW][y*lodw/CHUNKW][z*lodw/CHUNKW] += neighbor->sdf0[x][min(CHUNKW-1,y-CHUNKW)][min(CHUNKW-1,z-CHUNKW)];
		}
		}
		}
		neighbor = &get_chunk(chunk_position+v3s{1,1,1});
		for (s32 x = CHUNKW; x < lbounds.x*CHUNKW/lodw; ++x) {
		for (s32 y = CHUNKW; y < lbounds.y*CHUNKW/lodw; ++y) {
		for (s32 z = CHUNKW; z < lbounds.z*CHUNKW/lodw; ++z) {
			sdf_[x*lodw/CHUNKW][y*lodw/CHUNKW][z*lodw/CHUNKW] += neighbor->sdf0[min(CHUNKW-1,x-CHUNKW)][min(CHUNKW-1,y-CHUNKW)][min(CHUNKW-1,z-CHUNKW)];
		}
		}
		}
	}
#endif


	u8 edges[][2] {
		{0b000, 0b001},
		{0b010, 0b011},
		{0b100, 0b101},
		{0b110, 0b111},
		{0b000, 0b010},
		{0b001, 0b011},
		{0b100, 0b110},
		{0b101, 0b111},
		{0b000, 0b100},
		{0b001, 0b101},
		{0b010, 0b110},
		{0b011, 0b111},
	};

	v3f v[8] {
		{0, 0, 0},
		{0, 0, 1},
		{0, 1, 0},
		{0, 1, 1},
		{1, 0, 0},
		{1, 0, 1},
		{1, 1, 0},
		{1, 1, 1},
	};

	Vertex points[lodw+3][lodw+3][lodw+3];

	// DEBUG:
	for (auto &point : flatten(points)) {
		point = Vertex{
			.position = {0, 999, 0},
			.normal = {0, 1, 0},
		};
	}

	u32 edge_count = 0;

	{
		timed_block(profiler, profile_frame, "point generation");
		for (s32 lx = 0; lx < lbounds.x-1; ++lx) {
		for (s32 ly = 0; ly < lbounds.y-1; ++ly) {
		for (s32 lz = 0; lz < lbounds.z-1; ++lz) {
			s32 cx = lx * CHUNKW / lodw;
			s32 cy = ly * CHUNKW / lodw;
			s32 cz = lz * CHUNKW / lodw;

			s32 const o = CHUNKW / lodw;

			s32 d[8]{};

			d[0b000] = sdf(lx+0, ly+0, lz+0);
			d[0b001] = sdf(lx+0, ly+0, lz+1);
			d[0b010] = sdf(lx+0, ly+1, lz+0);
			d[0b011] = sdf(lx+0, ly+1, lz+1);
			d[0b100] = sdf(lx+1, ly+0, lz+0);
			d[0b101] = sdf(lx+1, ly+0, lz+1);
			d[0b110] = sdf(lx+1, ly+1, lz+0);
			d[0b111] = sdf(lx+1, ly+1, lz+1);

			u8 e =
				(u8)(d[0b000] > 0) +
				(u8)(d[0b001] > 0) +
				(u8)(d[0b010] > 0) +
				(u8)(d[0b011] > 0) +
				(u8)(d[0b100] > 0) +
				(u8)(d[0b101] > 0) +
				(u8)(d[0b110] > 0) +
				(u8)(d[0b111] > 0);

			if (e != 0 && e != 8) {
#if 0
				// no interpolation
				points[lx][ly][lz].position = V3f(cx, cy, cz) + 0.5f;
#else
				// with interpolation
				v3f point = {};
				f32 divisor = 0;

				for (auto &edge : edges) {
					auto a = edge[0];
					auto b = edge[1];
					if ((d[a] > 0) != (d[b] > 0)) {
						point += lerp(v[a], v[b], V3f((f32)d[a] / (d[a] - d[b])));
						divisor += 1;
					}
				}
				point /= divisor;
				points[lx][ly][lz].position = point * (CHUNKW/lodw) + V3f(cx,cy,cz);
#endif
				//points[lx][ly][lz].position.y -= 1 << (log2(CHUNKW) - log2(lodw));
#if 0
				points[lx][ly][lz].normal = -sdf_gradient(d, point);
#else
				// NOTE: normalize the normal so we can use it in vertex shader
				points[lx][ly][lz].normal = normalize((v3f)v3s{
					d[0b000] - d[0b100] +
					d[0b001] - d[0b101] +
					d[0b010] - d[0b110] +
					d[0b011] - d[0b111],
					d[0b000] - d[0b010] +
					d[0b001] - d[0b011] +
					d[0b100] - d[0b110] +
					d[0b101] - d[0b111],
					d[0b000] - d[0b001] +
					d[0b010] - d[0b011] +
					d[0b100] - d[0b101] +
					d[0b110] - d[0b111],
				});
#endif

				edge_count += 1;
			}
		}
		}
		}
	}

	StaticList<Vertex, lodw*lodw*lodw*6*3> vertices;

#if 0
	// points
	for (auto point : flatten(points)) {
		if (point.position.y != 999)
			vertices.add(point);
	}
#else
	// triangles
	{
		timed_block(profiler, profile_frame, "triangle generation");
		for (s32 lx = 1; lx < lbounds.x-1; ++lx) {
		for (s32 ly = 1; ly < lbounds.y-1; ++ly) {
		for (s32 lz = 1; lz < lbounds.z-1; ++lz) {
			s32 cx = lx * CHUNKW / lodw;
			s32 cy = ly * CHUNKW / lodw;
			s32 cz = lz * CHUNKW / lodw;

			s32 const o = CHUNKW / lodw;

			if ((sdf(lx,ly,lz) > 0) != (sdf(lx+1,ly,lz) > 0)) {
				auto _0 = points[lx][ly-1][lz-1]; // 0
				auto _1 = points[lx][ly+0][lz-1]; // 1
				auto _2 = points[lx][ly-1][lz+0]; // 2
				auto _3 = points[lx][ly+0][lz+0]; // 3

				if (!(sdf(lx,ly,lz) > 0)) {
					swap(_0, _3);
				}

				vertices.add(_0);
				vertices.add(_1);
				vertices.add(_2);
				vertices.add(_1);
				vertices.add(_3);
				vertices.add(_2);
			}
			if ((sdf(lx,ly,lz) > 0) != (sdf(lx,ly+1,lz) > 0)) {
				auto _0 = points[lx-1][ly][lz-1]; // 0
				auto _1 = points[lx+0][ly][lz-1]; // 1
				auto _2 = points[lx-1][ly][lz+0]; // 2
				auto _3 = points[lx+0][ly][lz+0]; // 3

				if (sdf(lx,ly,lz) > 0) {
					swap(_0, _3);
				}

				vertices.add(_0);
				vertices.add(_1);
				vertices.add(_2);
				vertices.add(_1);
				vertices.add(_3);
				vertices.add(_2);
			}
			if ((sdf(lx,ly,lz) > 0) != (sdf(lx,ly,lz+1) > 0)) {
				auto _0 = points[lx-1][ly-1][lz]; // 0
				auto _1 = points[lx+0][ly-1][lz]; // 1
				auto _2 = points[lx-1][ly+0][lz]; // 2
				auto _3 = points[lx+0][ly+0][lz]; // 3

				if (!(sdf(lx,ly,lz) > 0)) {
					swap(_0, _3);
				}

				vertices.add(_0);
				vertices.add(_1);
				vertices.add(_2);
				vertices.add(_1);
				vertices.add(_3);
				vertices.add(_2);
			}
		}
		}
		}
	}
#endif

	if (chunk.vertex_buffer) {
		chunk.vertex_buffer->Release();
		chunk.vertex_buffer = 0;
	}

	chunk.vertices.clear();

	if (!vertices.count)
		return;

	for (auto vertex : vertices)
		chunk.vertices.add(vertex.position);

	timed_block(profiler, profile_frame, "buffer generation");

	chunk.vert_count = vertices.count;

	ID3D11Buffer *vertex_buffer;
	defer { vertex_buffer->Release(); };
	{
		D3D11_BUFFER_DESC desc {
			.ByteWidth = (UINT)(sizeof(vertices[0]) * vertices.count),
			.Usage = D3D11_USAGE_DEFAULT,
			.BindFlags = D3D11_BIND_SHADER_RESOURCE,
			.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED,
			.StructureByteStride = sizeof(vertices[0]),
		};

		D3D11_SUBRESOURCE_DATA init {
			.pSysMem = vertices.data,
		};

		dhr(device->CreateBuffer(&desc, &init, &vertex_buffer));
	}
	{
		dhr(device->CreateShaderResourceView(vertex_buffer, 0, &chunk.vertex_buffer));
	}

	vertices.clear();

	chunk.time_since_remesh = 0;

	scoped_lock(chunks_with_meshes_mutex);
	chunks_with_meshes.insert(&chunk);
}

void update_chunk_mesh(Chunk &chunk, v3s chunk_position, u32 lodw) {
	scoped_lock(chunk.mutex);

	timed_function(profiler, profile_frame);

	switch (lodw) {
		case 1:   generate_chunk_lod<1  >(chunk, chunk_position); break;
		case 2:   generate_chunk_lod<2  >(chunk, chunk_position); break;
		case 4:   generate_chunk_lod<4  >(chunk, chunk_position); break;
		case 8:   generate_chunk_lod<8  >(chunk, chunk_position); break;
		case 16:  generate_chunk_lod<16 >(chunk, chunk_position); break;
		case 32:  generate_chunk_lod<32 >(chunk, chunk_position); break;
		// case 64:  generate_chunk_lod<64 >(chunk, chunk_position); break;
		default: invalid_code_path();
	}
}

void start() {

}

FrustumPlanes frustum;

bool chunk_in_view(v3s position) {
	auto relative_position = (v3f)((position-camera_position.chunk)*CHUNKW);
	return contains_sphere(frustum, relative_position+V3f(CHUNKW/2), sqrt3*CHUNKW);
}

extern "C" const Array<v3s8, pow3(DRAWD*2+1)> grid_map;

u32 chunks_remeshed_previous_frame;

u32 thread_count;

struct ChunkAndPosition {
	Chunk *chunk;
	v3s position;
};

#if 1
#define ITERATE_VISIBLE_CHUNKS_BEGIN \
	for (auto p_small : grid_map) { \
		auto p = (v3s)p_small;
#define ITERATE_VISIBLE_CHUNKS_END \
	}
#else
#define ITERATE_VISIBLE_CHUNKS_BEGIN \
	for (s32 x = -DRAWD; x <= DRAWD; ++x) { \
	for (s32 y = -DRAWD; y <= DRAWD; ++y) { \
	for (s32 z = -DRAWD; z <= DRAWD; ++z) { \
		v3s p = {x,y,z};
#define ITERATE_VISIBLE_CHUNKS_END \
	} \
	} \
	}
#endif

v3s r;
u32 iter_from_center_offset_sdf = 0;
u32 iter_from_center_offset_mesh = 0;

void generate_chunks_around() {
	timed_function(profiler, profile_frame);

	constexpr auto cell_threshold = pow3(8);

	auto avg_sdf_generation_seconds = total_sdfs_generated ? (f32)(total_time_wasted_on_generating_sdfs / total_sdfs_generated) / performance_frequency : 0.1f;
	// NOTE: this assumes that sdf generation takes 25% of a frame
	auto n_sdfs_can_generate_this_frame = max(1, floor_to_int(frame_time / avg_sdf_generation_seconds) * thread_count * 0.25f);
	s32 n_sdfs_generated = 0;

	auto work = make_work_queue(thread_pool);

	if (any_true(prev_camera_position.chunk != camera_position.chunk)) {
		timed_block(profiler, profile_frame, "remove chunks");
		iter_from_center_offset_sdf = 0;
		iter_from_center_offset_mesh = 0;
#if 1
		for (s32 x = prev_camera_position.chunk.x-DRAWD; x <= prev_camera_position.chunk.x+DRAWD; ++x) {
		for (s32 y = prev_camera_position.chunk.y-DRAWD; y <= prev_camera_position.chunk.y+DRAWD; ++y) {
		for (s32 z = prev_camera_position.chunk.z-DRAWD; z <= prev_camera_position.chunk.z+DRAWD; ++z) {
			if (camera_position.chunk.x-DRAWD <= x && x <= camera_position.chunk.x+DRAWD)
			if (camera_position.chunk.y-DRAWD <= y && y <= camera_position.chunk.y+DRAWD)
			if (camera_position.chunk.z-DRAWD <= z && z <= camera_position.chunk.z+DRAWD)
				continue;

			r = absolute(v3s{x,y,z} - camera_position.chunk);
			assert(r.x > DRAWD || r.y > DRAWD || r.z > DRAWD);
			auto &chunk = get_chunk(x,y,z);
			if (chunk.vertex_buffer) {
				chunk.vertex_buffer->Release();
				chunk.vertex_buffer = 0;
			}
			chunk.sdf_generated = false;
			chunk.lod_width = 0;
			chunk.vert_count = 0;
			chunk.neighbor_mask = {};
			find_and_erase_unordered(chunks_with_meshes, &chunk);
		}
		}
		}
#else
		auto before = aabb_center_radius(prev_camera_position.chunk, V3s(DRAWD));
		auto after  = aabb_center_radius(camera_position.chunk, V3s(DRAWD));

		auto left_volumes = subtract_points(before, after);

		print("XX: {}\n", left_volumes.count);

		for (auto volume : left_volumes) {
			print("  {} {}\n", volume.min, volume.max);
			for (s32 x = volume.min.x; x < volume.max.x; ++x) {
			for (s32 y = volume.min.y; y < volume.max.y; ++y) {
			for (s32 z = volume.min.z; z < volume.max.z; ++z) {
				assert(absolute(x - camera_position.chunk.x) > DRAWD);
				assert(absolute(y - camera_position.chunk.y) > DRAWD);
				assert(absolute(z - camera_position.chunk.z) > DRAWD);
				auto &chunk = get_chunk(x,y,z);
				if (chunk.vertex_buffer) {
					chunk.vertex_buffer->Release();
					chunk.vertex_buffer = 0;
				}

				chunk.sdf_generated = false;
				chunk.lod_width = 0;
			}
			}
			}
		}
#endif
	}
	{
		timed_block(profiler, profile_frame, "generate sdfs");
		bool reached_not_generated = false;
		for (auto pp_small = grid_map.begin() + iter_from_center_offset_sdf; pp_small != grid_map.end(); ++pp_small) {
			auto &p_small = *pp_small;
			auto p = (v3s)p_small;
			auto chunk_position = camera_position.chunk + p;
			auto &chunk = get_chunk(chunk_position);
			if (!chunk.sdf_generated) {
				if (!reached_not_generated) {
					reached_not_generated = true;
					iter_from_center_offset_sdf = pp_small - grid_map.data;
				}
				//if (chunk_in_view(chunk_position)) {
					work.push([chunk = &chunk, chunk_position]{
						generate_sdf(*chunk, chunk_position);
					});
					n_sdfs_generated += 1;
					if (n_sdfs_generated == n_sdfs_can_generate_this_frame) {
						goto _end;
					}
				//}
			}
		}
		_end:;
	}
	{
		timed_block(profiler, profile_frame, "remesh chunks");

		u32 cells[16] = {};

		auto remesh = [&] (Chunk *chunk, v3s chunk_position) {
			work.push([chunk, chunk_position, lod_width = chunk->lod_width] {
				update_chunk_mesh(*chunk, chunk_position, lod_width);
			});
			//print("remesh {}\n", chunk->position);
		};

		auto get_lodw_from_distance = [&](s32 distance) {
			for (s32 i = 1; i < CHUNKW; i *= 2)
				if (distance <= i)
					return CHUNKW/i;

			return 2;
		};


		u32 remesh_count = 0;

		u32 cells_ = 0;

		bool reached_not_generated = false;

		for (auto pp_small = grid_map.begin() + iter_from_center_offset_mesh; pp_small != grid_map.end(); ++pp_small) {
			auto &p_small = *pp_small;
			auto p = (v3s)p_small;
			auto chunk_position = camera_position.chunk + p;

			auto lodw = get_lodw_from_distance(max(absolute(p.x),absolute(p.y),absolute(p.z)));
			auto lod_index = log2(lodw);

			//if (visible)
			//	if (lodw == 32)
			//		debug_break();

			bool did_remesh = false;

			auto &chunk = get_chunk(chunk_position);

			//if (cells[lod_index] < cell_threshold) {
			if (chunk.sdf_generated) {
				auto new_neighbor_mask = get_neighbor_mask(chunk_position);
				if (chunk.neighbor_mask != new_neighbor_mask) {
					chunk.neighbor_mask = new_neighbor_mask;
					chunk.lod_width = lodw;
					remesh(&chunk, chunk_position);
					did_remesh = true;
				} else {
					if (chunk.lod_width < lodw) {
						if (chunk_in_view(chunk_position)) {
							chunk.lod_width = lodw;
							remesh(&chunk, chunk_position);
							did_remesh = true;
						}
					} else if (chunk.lod_width > lodw) {
						chunk.lod_width = lodw;
						remesh(&chunk, chunk_position);
						did_remesh = true;
					}
				}

					// if (chunk.lod_width < lodw) {
					// 	if (visible) {
					// 		chunk.lod_width = lodw;
					// 		remesh(&*chunk);
					// 		did_remesh = true;
					// 	}
					// } else if (chunk.lod_width > lodw) {
					// 	chunk.lod_width = lodw;
					// 	remesh(&*chunk);
					// 	did_remesh = true;
					// }

				if (did_remesh) {
					remesh_count++;
					//cells[lod_index] += chunk.lod_width*chunk.lod_width*chunk.lod_width;
					cells_ += chunk.lod_width*chunk.lod_width*chunk.lod_width;

					//bool may_do_more = false;
					//for (u32 i = 0; i < log2(CHUNKW); ++i) {
					//	may_do_more |= cells[i] < cell_threshold;
					//}
					//if (!may_do_more)
					//	goto _end2;

					if (cells_ >- cell_threshold) {
						goto _end2;
					}
				}
			} else {
				did_remesh = true;
			}

			bool has_full_mesh = chunk.sdf_generated && chunk.neighbor_mask.x && chunk.neighbor_mask.y && chunk.neighbor_mask.z;

			if (!has_full_mesh) {
				if (!reached_not_generated) {
					reached_not_generated = true;
					iter_from_center_offset_mesh = pp_small - grid_map.data;
				}
			}
		}
		_end2:
		//print("remesh_count: {}\n", remesh_count);
		chunks_remeshed_previous_frame = remesh_count;
	}

	{
		timed_block(profiler, profile_frame, "wait_for_completion");
		work.wait_for_completion();
	}
}

struct RayHit {
	bool did_hit;
	ChunkRelativePosition position;
	Chunk *chunk;
	f32 distance;
	v3f normal;

	explicit operator bool() const { return did_hit; }
};

RayHit global_raycast(ChunkRelativePosition origin_crp, v3f direction, f32 max_distance) {
	timed_function(profiler, profile_frame);

	RayHit result = {};

	if (length(direction) < 0.000001f)
		return {};
	direction = normalize(direction);

	auto end   = origin_crp + direction * max_distance;

	v3s _min = min(origin_crp.chunk, end.chunk) - 1;
	v3s _max = max(origin_crp.chunk, end.chunk) + 1;

	for (s32 x = _min.x; x <= _max.x; ++x) {
	for (s32 y = _min.y; y <= _max.y; ++y) {
	for (s32 z = _min.z; z <= _max.z; ++z) {
		v3s chunk_position = {x,y,z};
		auto &chunk = get_chunk(chunk_position);

		auto origin = origin_crp;
		origin.chunk -= chunk_position;

		auto ray = ray_origin_direction(origin.to_v3f(), direction);

		if (raycast(ray, aabb_min_max(v3f{}, V3f(CHUNKW+2)), true)) {
			for (u32 i = 0; i < chunk.vertices.count; i += 3) {
				auto a = chunk.vertices[i+0];
				auto b = chunk.vertices[i+1];
				auto c = chunk.vertices[i+2];

				if (auto hit = raycast(ray, triangle{a,b,c})) {
					if (!result || hit.distance < result.distance) {
						result = RayHit{
							.did_hit = true,
							.position = {
								.chunk = chunk_position,
								.local = hit.position
							},
							.chunk = &chunk,
							.distance = hit.distance,
							.normal = hit.normal,
						};
					}
				}
			}
		}
	}
	}
	}

	return result;
}

List<ChunkAndPosition> visible_chunks;

bool cursor_is_locked;
void lock_cursor() {
	cursor_is_locked = true;
	RECT rect;
	GetWindowRect(hwnd, &rect);

	rect = {
		.left   = (rect.left+rect.right)/2-1,
		.top    = (rect.top+rect.bottom)/2-1,
		.right  = (rect.left+rect.right)/2+1,
		.bottom = (rect.top+rect.bottom)/2+1,
	};

	ClipCursor(&rect);

	hide_cursor();
}
void unlock_cursor() {
	cursor_is_locked = false;
	ClipCursor(0);
	show_cursor();
}

ChunkRelativePosition cursor_position;

void update() {
	profile_frame = key_held('T');
	if (profile_frame) {
		profiler.reset();
	}
	defer {
		if (profile_frame)
			write_entire_file("frame.tmd"s, (Span<u8>)profiler.output_for_timed());
		profile_frame = false;
	};

	timed_function(profiler, profile_frame);

	if (key_down(Key_escape)) {
		if (cursor_is_locked)
			unlock_cursor();
		else
			lock_cursor();
	}

	if (key_down('H')) {
		camera_position = prev_camera_position = {0,1,0};
	}

	v3f camera_position_delta {
		key_held(Key_d) - key_held(Key_a),
		key_held(Key_e) - key_held(Key_q),
		key_held(Key_w) - key_held(Key_s),
	};

	camera_fov = clamp(camera_fov * powf(0.9f, mouse_wheel_delta), 1.f, 179.f);

	target_camera_angles.y += mouse_delta.x * camera_fov * 0.00005f;
	target_camera_angles.x += mouse_delta.y * camera_fov * 0.00005f;
	camera_angles = lerp(camera_angles, target_camera_angles, V3f(frame_time * 20));

	auto camera_rotation = m4::rotation_r_zxy(camera_angles);

	switch (camera_mode) {
		case CameraMode::fly: {
			prev_camera_position = camera_position;
			camera_position += camera_rotation * camera_position_delta * frame_time * 32 * (key_held(Key_shift) ? 10 : 1);

			if (key_down('F')) {
				camera_mode = CameraMode::walk;
				prev_camera_position = camera_position;
			}
			break;
		}
		case CameraMode::walk: {
			auto velocity = (camera_position - prev_camera_position).to_v3f();

			//f32 const max_sideways_velocity = 8*frame_time;
			//if (length(velocity.xz()) >= max_sideways_velocity) {
			//	auto xz = normalize(velocity.xz()) * max_sideways_velocity;
			//	velocity = {
			//		xz.x,
			//		velocity.y,
			//		xz.y,
			//	};
			//}

			prev_camera_position = camera_position;
			camera_position += velocity * v3f{.9,1,.9} + v3f{0,-9.8f,0}*pow2(frame_time);

			static v3f checkdir;

			f32 camera_height = 1.75f;
			if (auto collision = global_raycast(camera_position-v3f{0,camera_height,0}, (camera_position-prev_camera_position).to_v3f(), CHUNKW)) {
				if (collision.distance < 1) {
					checkdir = collision.normal;
				}
			}

			if (auto collision = global_raycast(camera_position-v3f{0,camera_height,0}, checkdir, CHUNKW)) {
				if (collision.distance < 2) {
					camera_position = collision.position + v3f{0,camera_height,0};
				}
			}

			camera_position += m4::rotation_r_y(camera_angles.y) * camera_position_delta * v3f{1,0,1} * frame_time * (key_held(Key_shift) ? 1 : 0.5f);

			if (key_down('F')) {
				camera_mode = CameraMode::fly;
			}
			break;
		}
	}

	auto rotproj = m4::perspective_left_handed((f32)window_client_size.x / window_client_size.y, radians(camera_fov), 0.1, CHUNKW * DRAWD) * m4::rotation_r_yxz(-camera_angles);
	auto camera_matrix = rotproj * m4::translation(-camera_position.local);

	frustum = create_frustum_planes_d3d(camera_matrix);

	// for_each_chunk([&](Chunk &chunk, v3s) {
	// 	chunk.time_since_remesh += frame_time;
	// });

	generate_chunks_around();

	if (mouse_held(0) || mouse_held(1)) {
		if (auto hit = global_raycast(camera_position, camera_rotation * v3f{0,0,1}, CHUNKW)) {
			timed_block(profiler, profile_frame, "apply deltas");

			cursor_position = hit.position;

			v3s center = round_to_int(hit.position.local);
			s32 const radius = 4;

			v3s cmin = hit.position.chunk + floor(center - radius-4, V3s(CHUNKW)) / CHUNKW;
			v3s cmax = hit.position.chunk + ceil (center + radius+2, V3s(CHUNKW)) / CHUNKW;

			auto add_sdf = [&](s32 x, s32 y, s32 z, s8 sdf) {
				if (sdf == 0)
					return;

				v3s c = hit.position.chunk;
				while (x < 0) { x += CHUNKW; c.x -= 1; }
				while (y < 0) { y += CHUNKW; c.y -= 1; }
				while (z < 0) { z += CHUNKW; c.z -= 1; }
				while (x >= CHUNKW) { x -= CHUNKW; c.x += 1; }
				while (y >= CHUNKW) { y -= CHUNKW; c.y += 1; }
				while (z >= CHUNKW) { z -= CHUNKW; c.z += 1; }

				auto &chunk = get_chunk(c);
				chunk.sdf0[x][y][z] = clamp(chunk.sdf0[x][y][z] + sdf, -128, 127);
			};

			for (s32 x = -radius; x <= radius; ++x) {
			for (s32 y = -radius; y <= radius; ++y) {
			for (s32 z = -radius; z <= radius; ++z) {
				auto c = center + v3s{x,y,z};
				s8 d = (1 - min(radius, length(V3f(x,y,z))) / radius) * 8 * (mouse_held(0) ? 1 : -1);
				add_sdf(c.x, c.y, c.z, d);
			}
			}
			}

			for (s32 x = cmin.x; x < cmax.x; ++x) {
			for (s32 y = cmin.y; y < cmax.y; ++y) {
			for (s32 z = cmin.z; z < cmax.z; ++z) {
				auto &chunk = get_chunk(x,y,z);
				filter_sdf(chunk);
			}
			}
			}

			auto work = make_work_queue(thread_pool);

			for (s32 x = cmin.x; x < cmax.x; ++x) {
			for (s32 y = cmin.y; y < cmax.y; ++y) {
			for (s32 z = cmin.z; z < cmax.z; ++z) {
				auto &chunk = get_chunk(x,y,z);
				work.push([chunk = &chunk, chunk_position = v3s{x,y,z}] {
					update_chunk_mesh(*chunk, chunk_position, chunk->lod_width);
				});
			}
			}
			}

			work.wait_for_completion();
		}
	}

	//if (key_down('G')) {
	//	for_each (chunks, [&](v3s chunk_position, Chunk &chunk) {
	//		print("{}: lodw: {}\n", chunk_position, chunk.lod_width);
	//	});
	//}

	//if (key_down('L')) {
	//	umm avg_count = 0;
	//	umm max_count = 0;
	//	umm min_count = ~0;
	//	for (auto &bucket : chunks.buckets) {
	//		auto count = count_of(bucket);
	//		avg_count += count;
	//		max_count = max(max_count, count);
	//		min_count = min(min_count, count);
	//	}
	//	print("avg: {}, min: {}, max: {}\n", (f32)avg_count / chunks.buckets.count, min_count, max_count);
	//}


	timed_block(profiler, profile_frame, "draw");

	immediate_context->VSSetConstantBuffers(0, 1, &frame_cbuffer.cbuffer);
	immediate_context->PSSetConstantBuffers(0, 1, &frame_cbuffer.cbuffer);
	immediate_context->VSSetConstantBuffers(1, 1, &chunk_cbuffer.cbuffer);
	immediate_context->PSSetConstantBuffers(1, 1, &chunk_cbuffer.cbuffer);

	//
	// SHADOWS
	//

	//frame_cbuffer.update({
	//	.mvp = camera_matrix,
	//	.rotproj = rotproj,
	//	.campos = camera_position.local,
	//	.ldir = normalize(v3f{1,3,2}),
	//});

	v3f light_angles = {pi/4, pi/6, 0};

	auto light_rotation = m4::rotation_r_zxy(light_angles);
	v3f light_dir = light_rotation * v3f{0,0,-1};

	f32 const shadow_pixels_in_meter = (f32)shadow_map_size / shadow_world_size / 2;

	auto lightr = m4::rotation_r_yxz(-light_angles);

	v3f lightpos = camera_position.local;
	lightpos = lightr * lightpos;
	lightpos *= shadow_pixels_in_meter;
	lightpos = round(lightpos);
	lightpos /= shadow_pixels_in_meter;
	lightpos = inverse(lightr) * lightpos;

	auto light_vp_matrix = m4::scale(1.f/shadow_world_size) * m4::rotation_r_yxz(-light_angles) * m4::translation(-lightpos);

	frame_cbuffer.update({
		//.mvp = m4::scale(.1f * v3f{(f32)window_client_size.x / window_client_size.y, 1, 1}) * m4::rotation_r_yxz(-v3f{pi}),
		.mvp = m4::translation(0,0,.5) * m4::scale(1,1,.5) * light_vp_matrix,
	});

	{
		D3D11_VIEWPORT viewport {
			.TopLeftX = 0,
			.TopLeftY = 0,
			.Width = shadow_map_size,
			.Height = shadow_map_size,
			.MinDepth = 0,
			.MaxDepth = 1,
		};
		immediate_context->RSSetViewports(1, &viewport);
	}

	immediate_context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
	immediate_context->VSSetShader(shadow_vs, 0, 0);
	immediate_context->PSSetShader(shadow_ps, 0, 0);
	immediate_context->RSSetState(0);
	immediate_context->OMSetRenderTargets(1, &shadow_rtv, shadow_dsv);
	immediate_context->OMSetBlendState(0, {}, -1);
	immediate_context->ClearRenderTargetView(shadow_rtv, v4f{}.s);
	immediate_context->ClearDepthStencilView(shadow_dsv, D3D11_CLEAR_DEPTH, 1, 0);
	for_each(chunks_with_meshes, [&](Chunk *chunk) {
		auto chunk_position = get_chunk_position(chunk);
		if (chunk->vertex_buffer) {
			auto relative_position = (v3f)((chunk_position-camera_position.chunk)*CHUNKW);
			chunk_cbuffer.update({
				.relative_position = relative_position,
				.actual_position = (v3f)(chunk_position*CHUNKW),
			});

			immediate_context->VSSetShaderResources(0, 1, &chunk->vertex_buffer);
			immediate_context->Draw(chunk->vert_count, 0);
		}
	});

	immediate_context->OMSetRenderTargets(0, 0, 0);
	immediate_context->PSSetShaderResources(3, 1, &shadow_srv);


	frame_cbuffer.update({
		.mvp = camera_matrix,
		.rotproj = rotproj,
		.light_vp_matrix = light_vp_matrix,
		.campos = camera_position.local,
		.ldir = light_dir,
	});

	{
		D3D11_VIEWPORT viewport {
			.TopLeftX = 0,
			.TopLeftY = 0,
			.Width = (f32)window_client_size.x,
			.Height = (f32)window_client_size.y,
			.MinDepth = 0,
			.MaxDepth = 1,
		};
		immediate_context->RSSetViewports(1, &viewport);
	}

	//
	// SKY RENDER
	//
	immediate_context->VSSetShader(sky_vs, 0, 0);
	immediate_context->PSSetShader(sky_ps, 0, 0);
	immediate_context->OMSetRenderTargets(1, &sky_rt, 0);
	immediate_context->Draw(36, 0);

	immediate_context->OMSetRenderTargets(1, &back_buffer, 0);
	immediate_context->ClearDepthStencilView(depth_stencil, D3D11_CLEAR_DEPTH, 1, 0);

	immediate_context->PSSetShaderResources(1, 1, &sky_srv);


	//
	// SKY BLIT
	//
	immediate_context->VSSetShader(blit_vs, 0, 0);
	immediate_context->PSSetShader(blit_ps, 0, 0);
	immediate_context->OMSetRenderTargets(1, &back_buffer, 0);
	immediate_context->Draw(6, 0);

	//
	// CHUNKS
	//
	immediate_context->OMSetRenderTargets(1, &back_buffer, depth_stencil);
	immediate_context->VSSetShader(chunk_vs, 0, 0);
	immediate_context->PSSetShader(chunk_solid_ps, 0, 0);

	visible_chunks.clear();

	for (auto &chunk : flatten(chunks)) {
		auto chunk_position = get_chunk_position(&chunk);
		if (chunk.vertex_buffer) {
			bool visible = chunk_in_view(chunk_position);
			if (visible) {
				visible_chunks.add({&chunk, chunk_position});
				auto relative_position = (v3f)((chunk_position-camera_position.chunk)*CHUNKW);
				chunk_cbuffer.update({
					.relative_position = relative_position,
					// .was_remeshed = (f32)(chunk.time_since_remesh < 0.1f),
					.actual_position = (v3f)(chunk_position*CHUNKW),
				});

				immediate_context->VSSetShaderResources(0, 1, &chunk.vertex_buffer);
				immediate_context->Draw(chunk.vert_count, 0);
			}
		}
	}

	if (key_down('R'))
		wireframe_rasterizer_enabled = !wireframe_rasterizer_enabled;

	if (wireframe_rasterizer_enabled) {
		immediate_context->RSSetState(wireframe_rasterizer);
		immediate_context->PSSetShader(chunk_wire_ps, 0, 0);
		immediate_context->OMSetBlendState(alpha_blend, {}, -1);
		for (auto chunk_and_position : visible_chunks) {
			auto chunk = chunk_and_position.chunk;
			auto chunk_position = chunk_and_position.position;
			auto relative_position = (v3f)((chunk_position-camera_position.chunk)*CHUNKW);
			chunk_cbuffer.update({
				.relative_position = relative_position,
				.actual_position = (v3f)(chunk_position*CHUNKW),
			});

			immediate_context->VSSetShaderResources(0, 1, &chunk->vertex_buffer);
			immediate_context->Draw(chunk->vert_count, 0);
		}
	}

	immediate_context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_LINELIST);

	immediate_context->VSSetShader(cursor_vs, 0, 0);
	immediate_context->PSSetShader(cursor_ps, 0, 0);
	chunk_cbuffer.update({
		.relative_position = cursor_position.local + (v3f)((cursor_position.chunk-camera_position.chunk)*CHUNKW),
	});
	immediate_context->Draw(6, 0);

	swap_chain->Present(0, 0);
}
void resize() {
	if (swap_chain) {
		if (back_buffer) {
			back_buffer->Release();
			depth_stencil->Release();
			sky_rt->Release();
		}

		dhr(swap_chain->ResizeBuffers(1, window_client_size.x, window_client_size.y, DXGI_FORMAT_UNKNOWN, 0));

		ID3D11Texture2D *back_buffer_texture = 0;
		dhr(swap_chain->GetBuffer(0, IID_PPV_ARGS(&back_buffer_texture)));
		defer { back_buffer_texture->Release(); };

		dhr(device->CreateRenderTargetView(back_buffer_texture, 0, &back_buffer));

		{
			ID3D11Texture2D *tex;
			{
				D3D11_TEXTURE2D_DESC desc {
					.Width = window_client_size.x,
					.Height = window_client_size.y,
					.MipLevels = 1,
					.ArraySize = 1,
					.Format = DXGI_FORMAT_D32_FLOAT,
					.SampleDesc = {1, 0},
					.Usage = D3D11_USAGE_DEFAULT,
					.BindFlags = D3D11_BIND_DEPTH_STENCIL,
				};
				dhr(device->CreateTexture2D(&desc, 0, &tex));
			}

			D3D11_DEPTH_STENCIL_VIEW_DESC desc {
				.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D,
				.Texture2D = {
					.MipSlice = 0
				}
			};
			dhr(device->CreateDepthStencilView(tex, &desc, &depth_stencil));
		}

		{
			ID3D11Texture2D *tex;
			{
				D3D11_TEXTURE2D_DESC desc {
					.Width = window_client_size.x,
					.Height = window_client_size.y,
					.MipLevels = 1,
					.ArraySize = 1,
					.Format = DXGI_FORMAT_R8G8B8A8_UNORM,
					.SampleDesc = {1, 0},
					.Usage = D3D11_USAGE_DEFAULT,
					.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET,
				};
				dhr(device->CreateTexture2D(&desc, 0, &tex));
			}
			{
				D3D11_RENDER_TARGET_VIEW_DESC desc {
					.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D,
					.Texture2D = {
						.MipSlice = 0
					}
				};
				dhr(device->CreateRenderTargetView(tex, &desc, &sky_rt));
			}
			{
				dhr(device->CreateShaderResourceView(tex, 0, &sky_srv));
			}
		}
	}
}

ID3D11VertexShader *create_vs(auto source) {
	ID3DBlob *errors = 0;
	auto print_errors = [&] {
		print(Span((char *)errors->GetBufferPointer(), errors->GetBufferSize()));
	};

	ID3DBlob *code = 0;
	if (FAILED(D3DCompile(source.data, source.count, "vs_source", 0, 0, "main", "vs_5_0", 0, 0, &code, &errors))) {
		print_errors();
		invalid_code_path();
	} else if (errors) {
		print_errors();
		errors->Release();
		errors = 0;
	}
	ID3D11VertexShader *vs;
	dhr(device->CreateVertexShader(code->GetBufferPointer(), code->GetBufferSize(), 0, &vs));
	return vs;
}

ID3D11PixelShader *create_ps(auto source) {
	ID3DBlob *errors = 0;
	auto print_errors = [&] {
		print(Span((char *)errors->GetBufferPointer(), errors->GetBufferSize()));
	};

	ID3DBlob *code = 0;
	if (FAILED(D3DCompile(source.data, source.count, "ps_source", 0, 0, "main", "ps_5_0", 0, 0, &code, &errors))) {
		print_errors();
		invalid_code_path();
	} else if (errors) {
		print_errors();
		errors->Release();
		errors = 0;
	}
	ID3D11PixelShader *ps;
	dhr(device->CreatePixelShader(code->GetBufferPointer(), code->GetBufferSize(), 0, &ps));
	return ps;
}

LRESULT WINAPI wnd_proc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
	switch (msg) {
		case WM_CREATE: {
			cursor_speed = get_cursor_speed();

			init_rawinput(RawInput_mouse);

			DXGI_SWAP_CHAIN_DESC sd {
				.BufferDesc = {
					.Width = 1,
					.Height = 1,
					.RefreshRate = {75, 1},
					.Format = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB,
				},
				.SampleDesc = {1, 0},
				.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT,
				.BufferCount = 1,
				.OutputWindow = hwnd,
				.Windowed = true,
			};

			auto feature = D3D_FEATURE_LEVEL_12_1;
			dhr(D3D11CreateDeviceAndSwapChain(0, D3D_DRIVER_TYPE_HARDWARE, 0, D3D11_CREATE_DEVICE_DEBUG, &feature, 1, D3D11_SDK_VERSION, &sd, &swap_chain, &device, 0, &immediate_context));
			dhr(device->QueryInterface(&debug_info_queue));

			chunk_vs = create_vs(FRAME_CBUFFER_STR R"(
cbuffer _ : register(b1) {
    float3 c_relative_position;
	float c_was_remeshed;
    float3 c_actual_position;
}

struct Vertex {
	float3 position;
	float3 normal;
	uint parent_vertex;
};

StructuredBuffer<Vertex> s_vertex_buffer : register(t0);

void main(
	in uint vertex_id : SV_VertexID,

	out float3 normal : NORMAL,
	out float3 color : COLOR,
	out float3 wpos : WPOS,
	out float3 view : VIEW,
	out float4 screen_uv : SCREEN_UV,
	out float4 shadow_map_uv : SHADOW_MAP_UV,
	out float4 position : SV_Position
) {
	float3 pos = s_vertex_buffer[vertex_id].position + c_relative_position;
	wpos = pos/8;
	normal = s_vertex_buffer[vertex_id].normal;
	view = pos - c_campos;

	if (c_was_remeshed == 0) {
#if 0
		// random color per chunk
		color = frac(c_actual_position * float3(1.23, 5.67, 8.9));
		color += color.zxy * float3(1.23, 5.67, 8.9);
		color += color.zxy * float3(1.23, 5.67, 8.9);
		color = frac(color+0.1f);
#else
		// ground color
		color = 1;
#endif
	} else {
		color = float3(1,0,0);
	}

	position = mul(c_mvp, float4(pos, 1.0f));
	screen_uv = position * float4(1, -1, 1, 1);

	shadow_map_uv = mul(light_vp_matrix, float4(pos, 1));
}
)"s);

			chunk_solid_ps = create_ps(FRAME_CBUFFER_STR R"(
SamplerState sam : register(s0);
SamplerComparisonState dsam : register(s1);
Texture2D skytex : register(t1);
Texture2D vorotex : register(t2);
Texture2D shadowtex : register(t3);

#define pow(t) \
t pow2(t v){return v*v;} \
t pow3(t v){return v*v*v;} \
t pow4(t v){return pow2(v*v);} \
t pow5(t v){return pow2(v*v)*v;}
pow(float)
pow(float2)
pow(float3)
pow(float4)
#undef pow

#define pi 3.1415926535897932384626433832795

float sdot(float3 a, float3 b) {
	return max(0,dot(a,b));
}

float map(float value, float source_min, float source_max, float dest_min, float dest_max) {
	return (value - source_min) / (source_max - source_min) * (dest_max - dest_min) + dest_min;
}
float map_clamped(float value, float source_min, float source_max, float dest_min, float dest_max) {
	return map(clamp(value, source_min, source_max), source_min, source_max, dest_min, dest_max);
}
float trowbridge_reitz_ggx(float NH, float roughness) {
	float r2 = pow2(roughness);
	return r2 / (pi * pow2(pow2(NH)*(r2 - 1) + 1));
}
float schlick_ggx(float NV, float k) {
	return NV / (NV * (1 - k) + k);
}
float k_direct(float roughness) {
	return pow2(roughness + 1) * 0.125f;
}
float k_ibl(float roughness) {
	return pow2(roughness) * 0.5f;
}
float smith_schlick(float NV, float NL, float k) {
	return schlick_ggx(NV, k) * schlick_ggx(NL, k);
}
float3 fresnel_schlick(float cos_theta, float3 F0) {
	return F0 + (1 - F0) * pow5(1 - cos_theta);
}
float3 cook_torrance(float D, float3 F, float G, float NV, float NL) {
	return D * F * G / (pi * NV * NL);
}


float4 triplanar(Texture2D tex, SamplerState sam, float3 wpos, float3 normal) {
	float4 pixel_color = 0;

	float3 t = abs(normal);
	// t = pow(t, 16);
	t *= 1.0f / (t.x + t.y + t.z);

	pixel_color += tex.Sample(sam, wpos.yz) * t.x;
	pixel_color += tex.Sample(sam, wpos.xz) * t.y;
	pixel_color += tex.Sample(sam, wpos.xy) * t.z;

	return pixel_color;
}

void main(
	in float3 normal : NORMAL,
	in float3 color_ : COLOR,
	in float3 wpos : WPOS,
	in float3 view : VIEW,
	in float4 screen_uv : SCREEN_UV,
	in float4 shadow_map_uv : SHADOW_MAP_UV,

	out float4 pixel_color : SV_Target
) {
	float3 L = c_ldir;
	float3 N = normalize(normal);
	float3 V = -normalize(view);
	float3 H = normalize(V + L);

	float NV = max(dot(N, V), 1e-3f);
	float NL = max(dot(N, L), 1e-3f);
	float NH = max(dot(N, H), 1e-3f);
	float VH = max(dot(V, H), 1e-3f);

	float trip = triplanar(vorotex, sam, wpos, normal);
	trip = map(trip, 0.1, 0.4, 0, 1);

	float3 grass = lerp(float3(.0,.1,.0), float3(.6,.9,.3), trip);
	float3 rock  = float3(.2,.2,.1) * trip;

	float gr = smoothstep(0.3, 0.7, normal.y);
	gr = lerp(gr, trip, (0.5f - abs(0.5f - gr)) * 2);

	float3 albedo = lerp(rock, grass, gr) * color_;

	float metalness = 0;
	float roughness = 1;

	screen_uv = (screen_uv / screen_uv.w) * 0.5 + 0.5;
	float3 ambient_color = skytex.Sample(sam, screen_uv.xy);

	float3 F0 = 0.04;
	F0 = lerp(F0, albedo, metalness);

	float D = trowbridge_reitz_ggx(NH, roughness);
	float G = smith_schlick(NV, NL, k_direct(roughness));
	float3 F = fresnel_schlick(NV, F0);
	float3 specular = cook_torrance(D, F, G, NV, NL);
	float3 diffuse = albedo * NL * (1 - metalness) / pi * (1 - specular);

	float3 ambient_specular = ambient_color * F * smith_schlick(NV, 1, k_ibl(roughness));
	float3 ambient_diffuse = albedo * ambient_color * (1-ambient_specular);
	float3 ambient = ambient_diffuse / pi + ambient_specular;


	shadow_map_uv /= shadow_map_uv.w;
	shadow_map_uv.y *= -1;
	float shadow_mask = saturate(map(length(shadow_map_uv.xyz), 0.9, 1, 0, 1));
	shadow_map_uv = shadow_map_uv * 0.5 + 0.5;

	float lightness = lerp(shadowtex.SampleCmpLevelZero(dsam, shadow_map_uv.xy, shadow_map_uv.z-0.01).x, 1, shadow_mask);
	pixel_color.rgb = ambient + (diffuse + specular) * lightness;
	pixel_color.a = 1;

	float fog = min(1, length(view) / ()" STRINGIZE(CHUNKW*DRAWD) R"());
	fog *= fog;
	pixel_color.rgb = lerp(pixel_color.rgb, ambient_color, fog);

	//pixel_color = shadowtex.SampleCmpLevelZero(dsam, shadow_map_uv.xy, shadow_map_uv.z-0.01);
	//pixel_color = shadowtex.Sample(sam, shadow_map_uv.xy);

}
)"s);


			chunk_wire_ps = create_ps(R"(
void main(in float3 normal : NORMAL, in float3 color : COLOR, out float4 pixel_color : SV_Target) {
	pixel_color.rgb = 0;
	pixel_color.a = 1;
}
)"s);

			cursor_vs = create_vs(R"(
cbuffer _ : register(b0) {
    float4x4 c_mvp;
    float4x4 c_rotproj;
	float3 c_campos;
}

cbuffer _ : register(b1) {
    float3 c_relative_position;
	float _;
    float3 c_actual_position;
}

void main(in uint vertex_id : SV_VertexID, out float3 color : COLOR, out float4 position : SV_Position) {
	float3 positions[] = {
		{-1, 0, 0},
		{ 1, 0, 0},
		{ 0,-1, 0},
		{ 0, 1, 0},
		{ 0, 0,-1},
		{ 0, 0, 1},
	};
	float3 colors[] = {
		{ 1, 0, 0},
		{ 1, 0, 0},
		{ 0, 1, 0},
		{ 0, 1, 0},
		{ 0, 0, 1},
		{ 0, 0, 1},
	};

	position = mul(c_mvp, float4(positions[vertex_id]+c_relative_position, 1.0f));
	color = colors[vertex_id];
}
)"s);
			cursor_ps = create_ps(R"(
void main(in float3 color: COLOR, out float4 pixel_color : SV_Target) {
	pixel_color = float4(color, 1);
}
)"s);

			sky_vs = create_vs(R"(
cbuffer _ : register(b0) {
    float4x4 c_mvp;
    float4x4 c_rotproj;
	float3 c_campos;
}

void main(in uint vertex_id : SV_VertexID, out float3 view : VIEW, out float4 position : SV_Position) {
	float3 positions[] = {
		// front
		{-1,-1, 1},
		{-1, 1, 1},
		{ 1,-1, 1},
		{-1, 1, 1},
		{ 1, 1, 1},
		{ 1,-1, 1},

		// back
		{-1, 1,-1},
		{-1,-1,-1},
		{ 1,-1,-1},
		{ 1, 1,-1},
		{-1, 1,-1},
		{ 1,-1,-1},

		// right
		{ 1,-1,-1},
		{ 1,-1, 1},
		{ 1, 1,-1},
		{ 1,-1, 1},
		{ 1, 1, 1},
		{ 1, 1,-1},

		// left
		{-1,-1, 1},
		{-1,-1,-1},
		{-1, 1,-1},
		{-1, 1, 1},
		{-1,-1, 1},
		{-1, 1,-1},

		// top
		{-1, 1, 1},
		{-1, 1,-1},
		{ 1, 1,-1},
		{ 1, 1, 1},
		{-1, 1, 1},
		{ 1, 1,-1},

		// bottom
		{-1,-1,-1},
		{-1,-1, 1},
		{ 1,-1,-1},
		{-1,-1, 1},
		{ 1,-1, 1},
		{ 1,-1,-1},
	};
	view = positions[vertex_id];
	position = mul(c_rotproj, float4(positions[vertex_id], 1));
}
)"s);
			sky_ps = create_ps(FRAME_CBUFFER_STR R"(
void main(in float3 view : VIEW, out float4 pixel_color : SV_Target) {
	float3 L = c_ldir;
	float3 V = normalize(view);
	float3 color = lerp(float3(.2,.4,.6), float3(1,1,.5), pow(dot(V, L)*.5+.5, 4));
	color += smoothstep(.99, 1, dot(V, L));
	// color = lerp(0, color, smoothstep(-0.5, 0, dot(V, float3(0,1,0))));
	pixel_color = float4(color, 1);
}
)"s);

			blit_vs = create_vs(R"(
void main(in uint vertex_id : SV_VertexID, out float2 uv : UV, out float4 position : SV_Position) {
	float2 positions[] = {
		{-1,-1},
		{-1, 1},
		{ 1,-1},
		{-1, 1},
		{ 1, 1},
		{ 1,-1},
	};
	uv = positions[vertex_id]*float2(0.5,-.5)+0.5;
	position = float4(positions[vertex_id], 0, 1);
}
)"s);
			blit_ps = create_ps(R"(
SamplerState sam : register(s0);
Texture2D skytex : register(t1);
void main(in float2 uv : UV, out float4 pixel_color : SV_Target) {
	pixel_color = skytex.Sample(sam, uv);
}
)"s);

			shadow_vs = create_vs(FRAME_CBUFFER_STR R"(
cbuffer _ : register(b1) {
    float3 c_relative_position;
	float c_was_remeshed;
    float3 c_actual_position;
}

struct Vertex {
	float3 position;
	float3 normal;
	uint parent_vertex;
};

StructuredBuffer<Vertex> s_vertex_buffer : register(t0);

void main(
	in uint vertex_id : SV_VertexID,
	out float4 screen_uv : SCREEN_UV,
	out float4 position : SV_Position
) {
	float3 pos = s_vertex_buffer[vertex_id].position + c_relative_position;
	position = mul(c_mvp, float4(pos, 1.0f));
	screen_uv = position * float4(1, -1, 1, 1);
}
)"s);

			shadow_ps = create_ps(FRAME_CBUFFER_STR R"(
float map(float value, float source_min, float source_max, float dest_min, float dest_max) {
	return (value - source_min) / (source_max - source_min) * (dest_max - dest_min) + dest_min;
}

void main(
	in float4 screen_uv : SCREEN_UV,
	out float4 pixel_color : SV_Target
) {
	//float dx = ddx(screen_uv.z);
	//float dy = ddy(screen_uv.z);
	//pixel_color = float2(screen_uv.z, dx*dx + dy*dy);//map(screen_uv.z, -1, 1, 0, 1);
	pixel_color = screen_uv.z;
}
)"s);


			frame_cbuffer.init();
			chunk_cbuffer.init();

			{
				D3D11_BLEND_DESC desc {
					.RenderTarget = {
						{
							.BlendEnable = true,
							.SrcBlend  = D3D11_BLEND_SRC_ALPHA,
							.DestBlend = D3D11_BLEND_INV_SRC_ALPHA,
							.BlendOp   = D3D11_BLEND_OP_ADD,
							.SrcBlendAlpha  = D3D11_BLEND_ZERO,
							.DestBlendAlpha = D3D11_BLEND_ZERO,
							.BlendOpAlpha   = D3D11_BLEND_OP_ADD,
							.RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL,
						}
					}
				};
				dhr(device->CreateBlendState(&desc, &alpha_blend));
			}
			{
				D3D11_RASTERIZER_DESC desc {
					.FillMode = D3D11_FILL_WIREFRAME,
					.CullMode = D3D11_CULL_BACK,
					.DepthBias = -32,
				};
				dhr(device->CreateRasterizerState(&desc, &wireframe_rasterizer));
			}

			{
				D3D11_SAMPLER_DESC desc {
					.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR,
					.AddressU = D3D11_TEXTURE_ADDRESS_WRAP,
					.AddressV = D3D11_TEXTURE_ADDRESS_WRAP,
					.AddressW = D3D11_TEXTURE_ADDRESS_WRAP,
					.MaxAnisotropy = 16,
					.MinLOD = 0,
					.MaxLOD = max_value<f32>,
				};

				ID3D11SamplerState *sampler;
				dhr(device->CreateSamplerState(&desc, &sampler));

				immediate_context->PSSetSamplers(0, 1, &sampler);
			}
			{
				D3D11_SAMPLER_DESC desc {
					.Filter = D3D11_FILTER_COMPARISON_MIN_MAG_MIP_LINEAR,
					.AddressU = D3D11_TEXTURE_ADDRESS_WRAP,
					.AddressV = D3D11_TEXTURE_ADDRESS_WRAP,
					.AddressW = D3D11_TEXTURE_ADDRESS_WRAP,
					.MaxAnisotropy = 16,
					.ComparisonFunc = D3D11_COMPARISON_LESS,
					.MinLOD = 0,
					.MaxLOD = max_value<f32>,
				};

				ID3D11SamplerState *sampler;
				dhr(device->CreateSamplerState(&desc, &sampler));

				immediate_context->PSSetSamplers(1, 1, &sampler);
			}

			{
				v4u8 pixels[256][256]{};
				for (s32 x = 0; x < 256; ++x) {
				for (s32 y = 0; y < 256; ++y) {
					s32 step_size = 4;
					for (s32 i = 0; i < 4; ++i) {
						v2s coordinate = {x,y};
						v2s scaled_tile = floor(coordinate, step_size);
						v2s tile_position = scaled_tile / step_size;
						v2f local_position = (v2f)(coordinate - scaled_tile) * reciprocal((f32)step_size);
						f32 min_distance_squared = 1000;

						static constexpr v2s offsets[] = {
							{-1,-1}, {-1, 0}, {-1, 1},
							{ 0,-1}, { 0, 0}, { 0, 1},
							{ 1,-1}, { 1, 0}, { 1, 1},
						};

						for (auto offset : offsets) {
							min_distance_squared = min(min_distance_squared, distance_squared(local_position, random_v2f(frac(tile_position + offset, V2s(256/step_size))) + (v2f)offset));
						}

						pixels[x][y] += (v4u8)V4u(sqrt(min_distance_squared) * voronoi_inv_largest_possible_distance_2d*255 / 4);

						step_size *= 2;
					}
				}
				}

				D3D11_TEXTURE2D_DESC desc {
					.Width = 256,
					.Height = 256,
					.MipLevels = 1,
					.ArraySize = 1,
					.Format = DXGI_FORMAT_R8G8B8A8_UNORM,
					.SampleDesc = {1,0},
					.BindFlags = D3D11_BIND_SHADER_RESOURCE,
				};

				D3D11_SUBRESOURCE_DATA data {
					.pSysMem = pixels,
					.SysMemPitch = 256*sizeof(v4u8),
				};

				ID3D11Texture2D *texture;
				dhr(device->CreateTexture2D(&desc, &data, &texture));

				ID3D11ShaderResourceView *grass_texture;
				dhr(device->CreateShaderResourceView(texture, 0, &grass_texture));

				immediate_context->PSSetShaderResources(2, 1, &grass_texture);
			}
			{
				ID3D11Texture2D *tex;
				{
					D3D11_TEXTURE2D_DESC desc {
						.Width = shadow_map_size,
						.Height = shadow_map_size,
						.MipLevels = 1,
						.ArraySize = 1,
						.Format = DXGI_FORMAT_D32_FLOAT,
						.SampleDesc = {1, 0},
						.Usage = D3D11_USAGE_DEFAULT,
						.BindFlags = D3D11_BIND_DEPTH_STENCIL,
					};
					dhr(device->CreateTexture2D(&desc, 0, &tex));
				}

				D3D11_DEPTH_STENCIL_VIEW_DESC desc {
					.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D,
					.Texture2D = {
						.MipSlice = 0
					}
				};
				dhr(device->CreateDepthStencilView(tex, &desc, &shadow_dsv));
			}

			{
				ID3D11Texture2D *tex;
				{
					D3D11_TEXTURE2D_DESC desc {
						.Width = shadow_map_size,
						.Height = shadow_map_size,
						.MipLevels = 1,
						.ArraySize = 1,
						.Format = DXGI_FORMAT_R32_FLOAT,
						.SampleDesc = {1, 0},
						.Usage = D3D11_USAGE_DEFAULT,
						.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET,
					};
					dhr(device->CreateTexture2D(&desc, 0, &tex));
				}
				{
					D3D11_RENDER_TARGET_VIEW_DESC desc {
						.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D,
						.Texture2D = {
							.MipSlice = 0
						}
					};
					dhr(device->CreateRenderTargetView(tex, &desc, &shadow_rtv));
				}
				{
					dhr(device->CreateShaderResourceView(tex, 0, &shadow_srv));
				}
			}
			break;
		}
		case WM_SETTINGCHANGE: {
			if (wp == SPI_SETMOUSESPEED) {
				cursor_speed = get_cursor_speed();
			}
			break;
		}
		case WM_DESTROY: {
			PostQuitMessage(0);
			return 0;
		}
		case WM_TIMER: {
			update();
			break;
		}
		case WM_ENTERSIZEMOVE: {
			SetTimer(hwnd, 0, 10, 0);
			break;
		}
		case WM_EXITSIZEMOVE: {
			KillTimer(hwnd, 0);
			break;
		}
		case WM_SIZE: {
			v2u new_size = {
				LOWORD(lp),
				HIWORD(lp),
			};

			if (!new_size.x || !new_size.y || (wp == SIZE_MINIMIZED))
				return 0;

			window_client_size = new_size;
			resize();
			return 0;
		}
	}
    return DefWindowProcW(hwnd, msg, wp, lp);
}


auto on_key_down = [](u8 key) {
	key_state[key] = KeyState_down | KeyState_repeated | KeyState_held;
};
auto on_key_repeat = [](u8 key) {
	key_state[key] |= KeyState_repeated;
};
auto on_key_up = [](u8 key) {
	key_state[key] = KeyState_up;
};
auto on_mouse_down = [](u8 button){
	key_state[256 + button] = KeyState_down | KeyState_held;
};
auto on_mouse_up = [](u8 button){
	key_state[256 + button] = KeyState_up;
};

int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE, LPWSTR, int) {
	init_allocator();

	_chunks = (decltype(_chunks))VirtualAlloc(0, sizeof(Chunk) * pow3(DRAWD*2+1), MEM_COMMIT|MEM_RESERVE, PAGE_READWRITE);
	for (auto &chunk : flatten(chunks)) {
		construct(chunk.vertices);
	}
	construct(profiler);
	construct(thread_pool);
	construct(chunks_with_meshes);
	construct(visible_chunks);
	thread_count = get_cpu_info().logical_processor_count;
	init_thread_pool(thread_pool, thread_count);
	//init_thread_pool(thread_pool, 0);

	AllocConsole();
	freopen("CONOUT$", "w", stdout);

	init_printer();

	make_os_timing_precise();

    WNDCLASSEXW c = {
        .cbSize = sizeof WNDCLASSEXW,
        .lpfnWndProc = wnd_proc,
        .hInstance = GetModuleHandleW(0),
        .hCursor = LoadCursor(0, IDC_ARROW),
        .lpszClassName = L"tclass",
    };

    assert_always(RegisterClassExW(&c));

    hwnd = CreateWindowExW(
		0, c.lpszClassName, L"tworld", WS_VISIBLE | WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT, 0, 0, c.hInstance, 0
	);

    assert_always(hwnd);

	prev_camera_position = camera_position;


	start();

	frame_time = 1 / 75.0f;

	lock_cursor();

	auto frame_time_counter = get_performance_counter();
	auto actual_frame_timer = create_precise_timer();

    while (1) {
		MSG message;
		mouse_delta = {};
		mouse_wheel_delta = 0;
		while (PeekMessageW(&message, 0, 0, 0, PM_REMOVE)) {
			bool mouse_went_down = false;
			bool mouse_went_up   = false;
			defer {
				if (mouse_went_down) {
					SetCapture(message.hwnd);
				}
				if (mouse_went_up) {
					ReleaseCapture();
				}
			};
			static bool alt_is_held;
			switch (message.message) {
				case WM_LBUTTONDOWN: { mouse_went_down = true; on_mouse_down(0); continue; }
				case WM_RBUTTONDOWN: { mouse_went_down = true; on_mouse_down(1); continue; }
				case WM_MBUTTONDOWN: { mouse_went_down = true; on_mouse_down(2); continue; }
				case WM_LBUTTONUP:   { mouse_went_up   = true; on_mouse_up  (0); continue; }
				case WM_RBUTTONUP:   { mouse_went_up   = true; on_mouse_up  (1); continue; }
				case WM_MBUTTONUP:   { mouse_went_up   = true; on_mouse_up  (2); continue; }
				case WM_INPUT: {
					RAWINPUT rawInput;
					if (UINT rawInputSize = sizeof(rawInput);
						GetRawInputData((HRAWINPUT)message.lParam, RID_INPUT, &rawInput, &rawInputSize,
										sizeof(RAWINPUTHEADER)) == -1) {
						invalid_code_path("Error: GetRawInputData");
					}
					if (rawInput.header.dwType == RIM_TYPEMOUSE) {
						auto &mouse = rawInput.data.mouse;

						mouse_delta += V2f(mouse.lLastX, mouse.lLastY) * cursor_speed;
					}
					continue;
				}
				case WM_SYSKEYDOWN:
				case WM_KEYDOWN: {
					u8 key = (u8)message.wParam;

					bool is_repeated = message.lParam & (LPARAM)(1 << 30);

					if (is_repeated) {
						on_key_repeat(key);
					} else {
						on_key_down(key);
						on_key_repeat(key);

						if (key == Key_alt) {
							alt_is_held = true;
						} else if (key == Key_f4) {
							if (alt_is_held) {
								return 0;
							}
						}
					}

					break;
				}
				case WM_SYSKEYUP:
				case WM_KEYUP: {
					u8 key = (u8)message.wParam;
					on_key_up(key);
					if (key == Key_alt) {
						alt_is_held = false;
					}
					continue;
				}
				case WM_MOUSEWHEEL: mouse_wheel_delta += (f32)GET_WHEEL_DELTA_WPARAM(message.wParam) / WHEEL_DELTA; continue;
				case WM_QUIT: return 0;
			}
			TranslateMessage(&message);
			DispatchMessageW(&message);
		}


		update();


		for (auto &state : key_state) {
			if (state & KeyState_down) {
				state &= ~KeyState_down;
			} else if (state & KeyState_up) {
				state = KeyState_none;
			}
			if (state & KeyState_repeated) {
				state &= ~KeyState_repeated;
			}
		}

		sync(frame_time_counter, frame_time);

		//print("{}\n", reset(actual_frame_timer)*1000);
		print("{}\n", iter_from_center_offset_mesh);
	}

    return 0;
}
