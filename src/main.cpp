#define _CRT_SECURE_NO_WARNINGS
#define _WINSOCK_DEPRECATED_NO_WARNINGS

#define NOMINMAX
#include <WS2tcpip.h>
#include <winsock2.h>
#undef assert

#define TL_IMPL
#define TL_DEBUG 1
#include <tl/common.h>
// this is stupid
#pragma push_macro("assert")
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
#include <tl/net.h>
#include <tl/masked_block_list.h>
#include <tl/mesh.h>
#include <tl/tracking_allocator.h>
#include <algorithm>
#include <random>

#pragma comment(lib, "freetype.lib")

using namespace tl;

#include <dxgi1_6.h>
#include <d3dcompiler.h>
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3dcompiler.lib")
#pragma comment(lib, "ws2_32.lib")

#include "d3d11.h"
#include "input.h"
#include "common.h"

#include <freetype/freetype.h>
#define TL_FONT_TEXTURE_HANDLE ID3D11ShaderResourceView *
#include <tl/font.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define USE_INDICES 0

// this is stupid
#pragma pop_macro("assert")

void print_wsa_error();

#define PERFECT_SDF 0

FontCollection *font_collection;

HWND hwnd;
IDXGISwapChain *swap_chain = 0;
ID3D11Device *device = 0;
ID3D11DeviceContext *immediate_context = 0;
Mutex immediate_context_mutex;

ID3D11RenderTargetView *back_buffer = 0;
ID3D11DepthStencilView *depth_stencil = 0;

ID3D11RenderTargetView   *sky_rt  = 0;
ID3D11ShaderResourceView *sky_srv = 0;

ID3D11DepthStencilView   *shadow_dsv = 0;
ID3D11ShaderResourceView *shadow_srv = 0;
u32 const shadow_map_size = 1024;
u32 const shadow_world_size = CHUNKW*4;

ID3D11InfoQueue* debug_info_queue = 0;

ID3D11RasterizerState *wireframe_rasterizer;
ID3D11RasterizerState *no_cull_rasterizer;
ID3D11RasterizerState *shadow_rasterizer;
ID3D11RasterizerState *no_cull_shadow_rasterizer;
bool wireframe_rasterizer_enabled;

ID3D11BlendState *alpha_blend;
ID3D11BlendState *font_blend;
ID3D11BlendState *alpha_to_coverage_blend;

ID3D11VertexShader *chunk_sdf_vs = 0;
ID3D11PixelShader *chunk_sdf_solid_ps = 0;
ID3D11PixelShader *chunk_sdf_wire_ps = 0;

ID3D11VertexShader *chunk_block_vs = 0;
ID3D11PixelShader  *chunk_block_ps = 0;

ID3D11VertexShader *cursor_vs = 0;
ID3D11PixelShader  *cursor_ps = 0;

ID3D11VertexShader *sky_vs = 0;
ID3D11PixelShader  *sky_ps = 0;

ID3D11VertexShader *blit_vs = 0;
ID3D11PixelShader  *blit_ps = 0;

ID3D11VertexShader *shadow_vs = 0;
ID3D11PixelShader  *shadow_ps = 0;

ID3D11VertexShader *font_vs = 0;
ID3D11PixelShader  *font_ps = 0;

ID3D11VertexShader *crosshair_vs = 0;
ID3D11PixelShader  *crosshair_ps = 0;

ID3D11GeometryShader *grass_gs = 0;
ID3D11VertexShader   *grass_vs = 0;
ID3D11PixelShader    *grass_ps = 0;

ID3D11VertexShader *tree_vs = 0;
ID3D11PixelShader  *tree_ps = 0;

ID3D11VertexShader *tree_shadow_vs = 0;
ID3D11PixelShader  *tree_shadow_ps = 0;

ID3D11ShaderResourceView *font_vertex_buffer = 0;

ID3D11ShaderResourceView *voronoi_albedo;
ID3D11ShaderResourceView *voronoi_normal;

ID3D11ShaderResourceView *planks_albedo;
ID3D11ShaderResourceView *planks_normal;

ID3D11ShaderResourceView *ground_albedo;
ID3D11ShaderResourceView *ground_normal;

ID3D11ShaderResourceView *grass_albedo;
ID3D11ShaderResourceView *grass_normal;

ID3D11SamplerState *default_sampler_wrap;
ID3D11SamplerState *default_sampler_clamp;

ID3D11ShaderResourceView *lod_mask;

struct VertexBuffer {
	ID3D11ShaderResourceView *view = 0;
	u32 vertex_count = 0;
};

VertexBuffer farlands_vb;

List<utf8> executable_path;
Span<utf8> executable_directory;

f32 frame_time;

u32 chunk_generation_amount_factor = 1;

bool draw_grass = true;
bool draw_trees = true;

struct Model {
	ID3D11ShaderResourceView *vb;
	ID3D11Buffer *ib;
	u32 index_count;

	ID3D11ShaderResourceView *albedo;
	ID3D11ShaderResourceView *normal;
	ID3D11ShaderResourceView *ao;
	bool no_cull = false;
};

struct LodList {
	struct Lod {
		Model model;
		u32 end_distance;
	};
	List<Lod> lods;
	Model &add_lod(u32 end_distance, Model model) {
		return lods.add({.model = model, .end_distance = end_distance}).model;
	}
	Model &add_lod(u32 end_distance) {
		return lods.add({.end_distance = end_distance}).model;
	}
	Model &get_lod(u32 distance) {
		for (auto &lod : lods) {
			if (distance <= lod.end_distance)
				return lod.model;
		}
		return lods.back().model;
	}
};

LodList tree_model;

struct TreeInstance {
	m3 matrix;
	v3f position;
};

struct FontVertex {
	v2f position;
	v2f uv;
};

template <class T>
struct StatValue {
	T current = {};
	T min = max_value<T>;
	T max = min_value<T>;
	T sum = {};
	u32 count = 0;

	void set(T value) {
		current = value;
		min = ::min(min, value);
		max = ::max(max, value);
		sum += value;
		count++;
	}
	void reset() {
		min = max_value<T>;
		max = min_value<T>;
		sum = {};
		count = 0;
	}

	T avg() { return count ? sum / count : T{}; }

	StatValue operator*(T that) {
		return {
			.current = current * that,
			.min     = min     * that,
			.max     = max     * that,
			.sum     = sum     * that,
		};
	}
};

template <class T>
inline umm append(StringBuilder &builder, StatValue<T> stat) {
	return append_format(builder, "{} | {} | {} | {}", stat.current, stat.min, stat.max, stat.avg());
}

StatValue<f32> actual_fps;
f32 smooth_fps;

Profiler profiler;
bool profile_frame;

f32 target_frame_time;
StatValue<f32> generate_time;

ThreadPool thread_pool;

s32 belt_item_index = 1;

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
		normalize();
		return *this;
	}
	ChunkRelativePosition &operator+=(ChunkRelativePosition that) {
		local += that.local;
		chunk += that.chunk;
		normalize();
		return *this;
	}

	ChunkRelativePosition operator+(v3f that) const { return ChunkRelativePosition(*this) += that; }
	ChunkRelativePosition operator-(v3f that) const { return ChunkRelativePosition(*this) -= that; }
	ChunkRelativePosition &operator-=(v3f that) { return *this += -that; }

	ChunkRelativePosition operator+(ChunkRelativePosition that) const { return ChunkRelativePosition(*this) += that; }
	ChunkRelativePosition operator-(ChunkRelativePosition that) const { return ChunkRelativePosition(*this) -= that; }
	ChunkRelativePosition &operator-=(ChunkRelativePosition that) { return *this += -that; }

	void normalize() {
		auto chunk_offset = floor(floor_to_int(local), CHUNKW);
		chunk += chunk_offset / CHUNKW;
		local -= (v3f)chunk_offset;
		ensure_valid();
	}
	void ensure_valid() {
		assert(!isnan(local.x));
		assert(!isnan(local.y));
		assert(!isnan(local.z));
	}
	v3f to_v3f() {
		return (v3f)(chunk * CHUNKW) + local;
	}
};

umm append(StringBuilder &builder, ChunkRelativePosition p) {
	return append_format(builder, "{}+{}", p.chunk, p.local);
}

ChunkRelativePosition opponent_position;

Optional<f32> sdf_at_interpolated(ChunkRelativePosition position);
Optional<v3f> gradient_at(ChunkRelativePosition position);
void add_sdf_sphere_synchronized(ChunkRelativePosition position, s32 brush_size, s32 strength);
void apply_physics(ChunkRelativePosition &position, ChunkRelativePosition &prev_position, v3f velocity_multiplier, v3f acceleration, aabb<v3f> collision_box);

union ComponentId {
	struct {
		u64 entity_index    : 24;
		u64 component_index : 24;
		u64 component_type  : 16;
	};
	u64 u;
};

struct Entity;

struct Component {
	Entity &entity;
	Component() : entity(*(Entity *)0) {}
	Component &operator=(Component const &that) {
		memcpy(this, &that, sizeof(that));
		return *this;
	}
	void on_destroy() {}
};

struct Entity {
	u32 entity_index;
	ChunkRelativePosition position = {};
	StaticList<ComponentId, 16> components;

	template <class T>
	T &add_component();
};

StaticMaskedBlockList<Entity, 256> entities;

LinearSet<Entity *> entities_to_remove;

Entity &create_entity() {
	auto added = entities.add();
	added.pointer->entity_index = added.index;
	return *added.pointer;
}

void enqueue_entity_removal(Entity &entity) {
	entities_to_remove.insert(&entity);
}


template <class T>
using ComponentList = StaticMaskedBlockList<T, 256>;

template <class T>
ComponentList<T> components;

template <class T>
extern u64 component_index;

struct PhysicsComponent : Component {
	ChunkRelativePosition prev_position = {};
	aabb<v3f> collision_box = {{-1,-1,-1}, {1,1,1}};
	v3f acceleration = {0,-9.8,0};
	v3f velocity_multiplier = {1,1,1};
	void update() {
		apply_physics(entity.position, prev_position, velocity_multiplier, acceleration, collision_box);
	}
	void set_velocity(v3f velocity) {
		prev_position = entity.position - velocity * target_frame_time;
	}
	void add_velocity(v3f velocity) {
		prev_position -= velocity * target_frame_time;
	}
};

struct ParticleSystemComponent : Component {
	struct Particle {
		v3f position;
		v3f velocity;
	};
	List<Particle> particles;

	v3s particles_chunk_position;
	f32 time_to_live_left = 1;
	void init(u32 particle_count) {
		particles_chunk_position = entity.position.chunk;
		particles.resize(particle_count);
		xorshift32 random { get_performance_counter() };
		for (auto &particle : particles) {
			particle.position = entity.position.local;
			particle.velocity = normalize(next_v3f(random) - 0.5f) * map<f32>(next_f32(random), 0, 1, 1, 2) * 10 * target_frame_time;
		}
	}
	void update() {
		for (auto &particle : particles) {
			particle.position += particle.velocity;
			particle.velocity += v3f{0,-9.8,0}*2*pow2(target_frame_time);
		}

		time_to_live_left -= target_frame_time;
		if (time_to_live_left < 0) {
			enqueue_entity_removal(entity);
		}
	}
	void on_destroy() {
		free(particles);
	}
};

struct GrenadeComponent : Component {
	f32 timer = 1;
	void update() {
		timer -= target_frame_time;
		if (timer < 0) {
			xorshift32 random { get_performance_counter() };

			for (u32 i = 0; i < 16; ++i) {
				add_sdf_sphere_synchronized(entity.position + (next_v3f(random) * 8 - 4), map<f32>(next_f32(random), 0, 1, 2, 4), -256);
			}

			enqueue_entity_removal(entity);

			auto &ps_entity = create_entity();
			ps_entity.position = entity.position;

			auto &ps = ps_entity.add_component<ParticleSystemComponent>();
			ps.time_to_live_left = 1;
			ps.init(100);
		}
	}
};


/*
#define e(name)
ENUMERATE_COMPONENTS
#undef e
*/
#define ENUMERATE_COMPONENTS \
e(PhysicsComponent) \
e(ParticleSystemComponent) \
e(GrenadeComponent) \

enum {
#define e(name) name##_index,
ENUMERATE_COMPONENTS
#undef e
};

#define e(name) template <> u64 component_index<name> = name##_index;
ENUMERATE_COMPONENTS
#undef e

template <class T>
T &Entity::add_component() {
	auto component = ::components<T>.add();

	*(Entity **)component.pointer = this;

	components.add({
		.entity_index = entity_index,
		.component_index = component.index,
		.component_type = component_index<T>,
	});

	return *component.pointer;
}

enum class CameraMode {
	walk,
	fly,
};

CameraMode camera_mode;

v3f target_camera_angles;
v3f camera_angles;

f32 camera_fov = 90;

Entity *camera;
ChunkRelativePosition camera_prev_position;
// NOTE: this is not the same as camera_prev_position.chunk
// camera_prev_position is used in physics, so it may not represent the actual previous position in previous frame.
v3s camera_chunk_last_frame;

struct NeighborMask {
	bool x : 1;
	bool y : 1;
	bool z : 1;
	auto operator<=>(NeighborMask const &) const = default;
};
umm append(StringBuilder &builder, NeighborMask v) {
	return append_format(builder, "{{{}, {}, {}}}", v.x, v.y, v.z);
}

struct ChunkVertex {
	v3f position;
	u32 normal;
};

u32 encode_normal(v3f n) {
	f32 const e = 0.001;
	return
		((u8)map<f32>(n.x, -1, 1, 0, 256-e) << 0) |
		((u8)map<f32>(n.y, -1, 1, 0, 256-e) << 8) |
		((u8)map<f32>(n.z, -1, 1, 0, 256-e) << 16);
}

v3f decode_normal(u32 n) {
	return {
		map<f32>((n >>  0) & 0xff, 0, 256, -1, 1),
		map<f32>((n >>  8) & 0xff, 0, 256, -1, 1),
		map<f32>((n >> 16) & 0xff, 0, 256, -1, 1),
	};
}

umm append(StringBuilder &builder, VertexBuffer v) {
	return append_format(builder, "{{view={}, count={}}}", v.view, v.vertex_count);
}


struct Block {
	bool solid : 1;
};

#define DEBUG_CHUNK_THREAD_ACCESS 0

struct Sdf {
	s8 _0[CHUNKW][CHUNKW][CHUNKW];
	s8 _1[CHUNKW/2][CHUNKW/2][CHUNKW/2];
	s8 _2[CHUNKW/4][CHUNKW/4][CHUNKW/4];
	s8 _3[CHUNKW/8][CHUNKW/8][CHUNKW/8];
	s8 _4[CHUNKW/16][CHUNKW/16][CHUNKW/16];
	s8 _5[CHUNKW/32][CHUNKW/32][CHUNKW/32];
};

struct Chunk {
	ID3D11ShaderResourceView *sdf_vb = 0;
	u32 sdf_vb_vertex_count[6] = {};
	u32 sdf_vb_vertex_offset[6] = {};
	VertexBuffer grass_vb = {};
	VertexBuffer blocks_vb = {};
	SBuffer<TreeInstance> trees_instances_buffer;

#if USE_INDICES
	ID3D11Buffer *index_buffer = 0;
#endif

	List<v3f> sdf_vertex_positions;
#if USE_INDICES
	List<u32> indices;
	u32 indices_count = 0;
#else
#endif

	Sdf *sdf = 0;

	Block (*blocks)[CHUNKW][CHUNKW][CHUNKW] = 0;

	f32 lod_t = 1;

	u8 lod_previous_frame = 0;
	u8 previous_lod = 0;
	u8 frames_since_remesh = 0;
	s8 average_sdf = 0;
	NeighborMask neighbor_mask = {};
	bool sdf_generated           : 1 = false;
	bool sdf_mesh_generated      : 1 = false;
	bool block_mesh_generated    : 1 = false;
	bool has_surface             : 1 = false;
	bool needs_saving            : 1 = false;
	bool needs_filter_and_remesh : 1 = false;

#if DEBUG_CHUNK_THREAD_ACCESS
	RecursiveMutex mutex;
#endif

	Chunk() = default;
	Chunk(Chunk const &) = delete;
	Chunk(Chunk &&) = delete;
	Chunk &operator==(Chunk const &) = delete;
	Chunk &operator==(Chunk &&) = delete;
};

Chunk  (* _chunks)[DRAWD*2+1][DRAWD*2+1][DRAWD*2+1];

#define chunks  (*_chunks)

int _1_ = sizeof Chunk;

void allocate_sdf(Chunk *chunk) {
	assert(!chunk->sdf);
	chunk->sdf = (decltype(chunk->sdf))VirtualAlloc(0, sizeof(*chunk->sdf), MEM_RESERVE|MEM_COMMIT, PAGE_READWRITE);
	assert(chunk->sdf);
}

void free_sdf(Chunk *chunk) {
	VirtualFree(chunk->sdf, 0, MEM_FREE);
	chunk->sdf = 0;
}

void allocate_blocks(Chunk *chunk) {
	assert(!chunk->blocks);
	chunk->blocks = (decltype(chunk->blocks))VirtualAlloc(0, sizeof(*chunk->blocks), MEM_RESERVE|MEM_COMMIT, PAGE_READWRITE);
	assert(chunk->blocks);
}

void free_blocks(Chunk *chunk) {
	VirtualFree(chunk->blocks, 0, MEM_FREE);
	chunk->blocks = 0;
}

// These two functions define axis ordering.
// NOTE: x,y,z ordering may be not the best.
// For good masking there need to be big portions of empty chunks.
// Because the world is relatively flat, i think the best option would
// be to index by y,x,z or y,z,x.
// NOTE: I tested xyz vs yxz and it does look like yxz is better, but not by much:
// 440 masks skipped by yxz, 427 by xyz.
Chunk *get_chunk(v3s v) {
	s32 const s = DRAWD*2+1;
	v = frac(v, s);
	return &chunks[v.y][v.x][v.z];
}
v3s get_chunk_position(s32 index) {
	s32 const s = DRAWD*2+1;

	auto v = v3s {
		(index / s),
		(index / (s * s)),
		index,
	} % s;

	v += floor(camera->position.chunk+DRAWD, s);

	v -= s * (v3s)(v > camera->position.chunk+DRAWD);

	auto r = absolute(v - camera->position.chunk);
	assert(-DRAWD <= r.x && r.x <= DRAWD);
	assert(-DRAWD <= r.y && r.y <= DRAWD);
	assert(-DRAWD <= r.z && r.z <= DRAWD);

	return v;
}




Chunk *get_chunk(s32 x, s32 y, s32 z) {
	return get_chunk({x, y, z});
}

// (0 4) (1 4)|(2 4) (3 4) (4 4)
// (0 3) (1 3)|(2 3) (3 3) <4 3>
// (0 2) (1 2)|(2 2) (3 2) (4 2)
// (0 1) (1 1)|(2 1) (3 1) (4 1)
// -----------------------------
// (0 0) (1 0)|(2 0) (3 0) (4 0)

// camera_chunk = (4 3)


u32 get_chunk_index(Chunk *chunk) { return (u32)(chunk - &chunks[0][0][0]); }
u32 get_chunk_index(v3s position) { return get_chunk_index(get_chunk(position)); }

v3s get_chunk_position(Chunk *chunk) {
	return get_chunk_position(get_chunk_index(chunk));
}

//
// Mark empty chunks in packed bitmask for faster iteration.
// Fully empty chunks will have 0 bit in corresponding index.
// When drawing, if the whole u64 is zero, we can skip checking these 64 chunks and go to the next 64 immediately.
// This thing reduced shadows block time from ~14.5 down to ~10.5 milliseconds on DRAWD of 16 .
//
u32 const bits_in_chunk_mask = 64;
u64 nonempty_chunk_mask[ceil((u32)pow3(DRAWD*2+1), bits_in_chunk_mask) / bits_in_chunk_mask];
Mutex nonempty_chunk_mask_mutex;

void update_chunk_mask(Chunk *chunk, v3s position) {
	scoped_lock(nonempty_chunk_mask_mutex);


	auto set_chunk_mask = [&](v3s position, bool new_mask) {
		auto index = get_chunk_index(position);

		// assert(index / bits_in_chunk_mask < count_of(nonempty_chunk_mask));
		auto &mask = nonempty_chunk_mask[index / bits_in_chunk_mask];
		auto bit = (u64)1 << (index % bits_in_chunk_mask);
		if (new_mask)
			mask |= bit;
		else
			mask &= ~bit;
	};

	// NOTE:
	// Instead of checking every sdf_vb_vertex_count maybe just use ->sdf ?
	set_chunk_mask(
		position,
		chunk->blocks_vb.vertex_count ||
		chunk->sdf_vb_vertex_count[0] ||
		chunk->sdf_vb_vertex_count[1] ||
		chunk->sdf_vb_vertex_count[2] ||
		chunk->sdf_vb_vertex_count[3] ||
		chunk->sdf_vb_vertex_count[4] ||
		chunk->sdf_vb_vertex_count[5] ||
		chunk->trees_instances_buffer.count
	);
}


void apply_physics(ChunkRelativePosition &position, ChunkRelativePosition &prev_position, v3f velocity_multiplier, v3f acceleration, aabb<v3f> collision_box) {

	auto velocity = (position - prev_position).to_v3f();

	prev_position = position;
	position += velocity*velocity_multiplier + acceleration*2*pow2(target_frame_time);


	v3f collision_points[] = {
		{collision_box.min.x,collision_box.min.y,collision_box.min.z},
		{collision_box.min.x,collision_box.min.y,collision_box.max.z},
		{collision_box.min.x,collision_box.max.y,collision_box.min.z},
		{collision_box.min.x,collision_box.max.y,collision_box.max.z},
		{collision_box.max.x,collision_box.min.y,collision_box.min.z},
		{collision_box.max.x,collision_box.min.y,collision_box.max.z},
		{collision_box.max.x,collision_box.max.y,collision_box.min.z},
		{collision_box.max.x,collision_box.max.y,collision_box.max.z},
	};

	bool fallback = false;
	for (auto collision_point : collision_points) {
		auto corner_pos = position + collision_point;

		s32 const iter_max = 256;

		s32 i = 0;
		for (i = 0; i < iter_max; ++i) {
			if (auto sdf_ = sdf_at_interpolated(corner_pos)) {
				auto sdf = sdf_.value_unchecked();
				if (sdf < 0) {
					break;
				}
				if (auto gradient = gradient_at(corner_pos)) {
					corner_pos += gradient.value_unchecked() / iter_max;
				} else {
					corner_pos += v3f{0,target_frame_time,0};
					fallback = true;
				}
			}
		}
		if (i == iter_max)
			fallback = true;
		position = corner_pos - collision_point;
	}
	if (fallback) {
		prev_position = position;
	} else {
		auto local_min = floor_to_int(position.local + collision_box.min);
		auto local_max =  ceil_to_int(position.local + collision_box.max);
		auto fmin = floor(local_min, CHUNKW) / CHUNKW;
		auto fmax = floor(local_max, CHUNKW) / CHUNKW;
		auto cmin = position.chunk + fmin;
		auto cmax = position.chunk + fmax;

		for (s32 cx = cmin.x; cx <= cmax.x; ++cx) {
		for (s32 cy = cmin.y; cy <= cmax.y; ++cy) {
		for (s32 cz = cmin.z; cz <= cmax.z; ++cz) {
			auto chunk = get_chunk(cx,cy,cz);
			if (!chunk->sdf_generated)
				continue;

			auto coff = (v3s{cx,cy,cz} - position.chunk) * CHUNKW;
			auto bmin = max(local_min - coff, V3s(0));
			auto bmax = min(local_max - coff, V3s(CHUNKW-1));

			for (s32 bx = bmin.x; bx <= bmax.x; ++bx) {
			for (s32 by = bmin.y; by <= bmax.y; ++by) {
			for (s32 bz = bmin.z; bz <= bmax.z; ++bz) {
				if (!chunk->blocks)
					continue;
				if ((*chunk->blocks)[bx][by][bz].solid) {
					auto box = aabb_min_size((v3f)v3s{bx,by,bz} + (v3f)coff, V3f(1));
					box.min -= collision_box.max;
					box.max -= collision_box.min;

					auto p = position.local;
					auto dir = (position - prev_position).to_v3f();

					if (in_bounds(p, box)) {
						defer {
							position.local = p;
							position.normalize();
						};
						if (length(dir) > 0.000001) {
							if (auto hit = raycast(ray_origin_end(p-dir,p), box)) {
								p += project(hit.position - p, hit.normal) + hit.normal * 0.001f;
								continue;
							}
						}

						f32 s[] {
							absolute(p.x - box.min.x) * (dot(dir, v3f{ 1, 0, 0}) * 0.5f + 0.5f),
							absolute(p.x - box.max.x) * (dot(dir, v3f{-1, 0, 0}) * 0.5f + 0.5f),
							absolute(p.y - box.min.y) * (dot(dir, v3f{ 0, 1, 0}) * 0.5f + 0.5f),
							absolute(p.y - box.max.y) * (dot(dir, v3f{ 0,-1, 0}) * 0.5f + 0.5f),
							absolute(p.z - box.min.z) * (dot(dir, v3f{ 0, 0, 1}) * 0.5f + 0.5f),
							absolute(p.z - box.max.z) * (dot(dir, v3f{ 0, 0,-1}) * 0.5f + 0.5f),
						};

						auto m = min(s);

						     if (m == s[0]) p.x = box.min.x;
						else if (m == s[1]) p.x = box.max.x;
						else if (m == s[2]) p.y = box.min.y;
						else if (m == s[3]) p.y = box.max.y;
						else if (m == s[4]) p.z = box.min.z;
						else                p.z = box.max.z;

					}
				}
			}
			}
			}
		}
		}
		}
	}
}

template <class Fn>
void for_each_chunk(Fn &&fn) {
	timed_function(profiler, profile_frame);

	for (s32 x = camera->position.chunk.x-DRAWD; x <= camera->position.chunk.x+DRAWD; ++x) {
	for (s32 y = camera->position.chunk.y-DRAWD; y <= camera->position.chunk.y+DRAWD; ++y) {
	for (s32 z = camera->position.chunk.z-DRAWD; z <= camera->position.chunk.z+DRAWD; ++z) {
		fn(get_chunk(x,y,z), {x,y,z});
	}
	}
	}
}

NeighborMask get_neighbor_mask(v3s position) {
	auto _100 = get_chunk(position + v3s{1,0,0})->sdf_generated;
	auto _010 = get_chunk(position + v3s{0,1,0})->sdf_generated;
	auto _001 = get_chunk(position + v3s{0,0,1})->sdf_generated;
	auto _011 = get_chunk(position + v3s{0,1,1})->sdf_generated;
	auto _101 = get_chunk(position + v3s{1,0,1})->sdf_generated;
	auto _110 = get_chunk(position + v3s{1,1,0})->sdf_generated;
	auto _111 = get_chunk(position + v3s{1,1,1})->sdf_generated;

	NeighborMask mask;

	if (
		_100 &&
		_010 &&
		_001 &&
		_011 &&
		_101 &&
		_110 &&
		_111
	) mask = {.x=true,.y=true,.z=true};
	else if (
		_100 &&
		_001 &&
		_101
	) mask = {.x=true,.z=true};

	else if (
		_100 &&
		_010 &&
		_110
	) mask = {.x=true,.y=true};

	else if (
		_010 &&
		_001 &&
		_011
	) mask = {.y=true,.z=true};

	else if (_100) mask = {.x=true};
	else if (_010) mask = {.y=true};
	else if (_001) mask = {.z=true};
	else mask = {};

	auto r = position != camera->position.chunk + DRAWD;
	mask.x &= r.x;
	mask.y &= r.y;
	mask.z &= r.z;
	return mask;
}

template <>
inline static u64 get_hash(v3s const &a) {
	auto const s = DRAWD*2+1;
	return
		a.x * s * s +
		a.y * s +
		a.z;
}

struct V3sHashTraits : DefaultHashTraits<v3s> {
	inline static bool are_equal(v3s a, v3s b) {
		return
			(a.x == b.x) &
			(a.y == b.y) &
			(a.z == b.z);
	}
};

struct SavedChunk {
	s8 sdf[CHUNKW][CHUNKW][CHUNKW];
	Block blocks[CHUNKW][CHUNKW][CHUNKW];
};

HashMap<v3s, SavedChunk, V3sHashTraits> saved_chunks;

void save_chunk(Chunk *chunk, v3s chunk_position) {
	auto &saved = saved_chunks.get_or_insert(chunk_position);
	if (chunk->sdf) memcpy(saved.sdf, chunk->sdf->_0,     sizeof(saved.sdf));
	else            memset(saved.sdf, chunk->average_sdf, sizeof(saved.sdf));

	if (chunk->blocks) memcpy(saved.blocks, *chunk->blocks, sizeof(saved.blocks));
	else               memset(saved.blocks, 0,              sizeof(saved.blocks));
}


// NOTE: in seconds
f32 smoothed_average_generation_time;

u64 sdfs_generated_per_frame;

s32x8 randomize(s32x8 v) {
	v = vec32_xor(v, s32x8_set1(0x55555555u));
	v = s32x8_mul(v, s32x8_set1(u32_random_primes[0]));
	// v = vec32_xor(v, s32x8_set1(0x33333333u));
	// v = s32x8_mul(v, s32x8_set1(u32_random_primes[1]));
	return v;
}

f32x8 random_f32x8(s32x8 x, s32x8 y, s32x8 z) {
	x = randomize(x);
	x = randomize(vec32_xor(x, y));
	x = randomize(vec32_xor(x, z));
	return f32x8_mul(s32x8_to_f32x8(s32x8_slri(x, 8)), f32x8_set1(1.0f / ((1 << 24) - 1)));
}

void filter_sdf(Chunk *chunk) {
	timed_function(profiler, profile_frame);

	assert(chunk->sdf);

#define LOD(div, dst, src) \
	for (s32 sx = 0, dx = 0; sx < CHUNKW/div; sx += 2, dx += 1) { \
	for (s32 sy = 0, dy = 0; sy < CHUNKW/div; sy += 2, dy += 1) { \
	for (s32 sz = 0, dz = 0; sz < CHUNKW/div; sz += 2, dz += 1) { \
		chunk->sdf->dst[dx][dy][dz] = ( \
			chunk->sdf->src[sx+0][sy+0][sz+0] + \
			chunk->sdf->src[sx+0][sy+0][sz+1] + \
			chunk->sdf->src[sx+0][sy+1][sz+0] + \
			chunk->sdf->src[sx+0][sy+1][sz+1] + \
			chunk->sdf->src[sx+1][sy+0][sz+0] + \
			chunk->sdf->src[sx+1][sy+0][sz+1] + \
			chunk->sdf->src[sx+1][sy+1][sz+0] + \
			chunk->sdf->src[sx+1][sy+1][sz+1] \
			) >> 3; \
	} \
	} \
	}

	LOD(1,  _1, _0);
	LOD(2,  _2, _1);
	LOD(4,  _3, _2);
	LOD(8,  _4, _3);
	LOD(16, _5, _4);

#undef LOD
}

// Generates 8 sdf values at [x,y,z] to (x+1,y+1,z+8)
f32x8 sdf_func_x8(s32 x, s32 y, s32 z, s32 coord_scale = 1) {
	f32x8 d = {};
	s32 scale = 1;

	s32x8 gx = s32x8_set1(x);
	s32x8 gy = s32x8_set1(y);
	s32x8 gz = s32x8_add(s32x8_set1(z), s32x8_mul(s32x8_set1(coord_scale), s32x8_set(0,1,2,3,4,5,6,7)));

	f32 scale_sum = 0;

	for (s32 i = 0; i < 4; ++i) {
		s32 step = scale*4;

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
		h = f32x8_mul(f32x8_sub(h, f32x8_set1(0.5f)), f32x8_set1(scale));
		//h = f32x8_mul(f32x8_sub(h, f32x8_mul(s32x8_to_f32x8(gy), f32x8_set1(8))), f32x8_set1(scale));
		//d = f32x8_add(d, f32x8_add(f32x8_set1(-0.5f), f32x8_mul(s32x8_to_f32x8(s32x8_sub(s32x8_set1(CHUNKW/2), gy)), f32x8_set1(0.0125f))));
		d = f32x8_add(d, h);

		scale_sum += scale;
		scale *= 8;
	}

	d = f32x8_div(d, f32x8_set1(scale_sum));

	d = f32x8_add(f32x8_add(d, f32x8_mul(s32x8_to_f32x8(gy), f32x8_set1(-0.125f))), f32x8_set1(100));
	return d;
}
f32x8 sdf_func_x8(v3s v, s32 coord_scale = 1) {
	return sdf_func_x8(v.x, v.y, v.z, coord_scale);
}

void generate_sdf(Chunk *chunk, v3s chunk_position) {
	timed_function(profiler, profile_frame);

	assert(!chunk->sdf);

	defer {
		chunk->sdf_generated = true;
		atomic_increment(&sdfs_generated_per_frame);
	};
#if 0
	if (chunk_position.y > 8) {
		memset(chunk.sdf0, -128, sizeof chunk.sdf0);
		memset(chunk.sdf1, -128, sizeof chunk.sdf1);
		memset(chunk.sdf2, -128, sizeof chunk.sdf2);
		memset(chunk.sdf3, -128, sizeof chunk.sdf3);
		memset(chunk.sdf4, -128, sizeof chunk.sdf4);
		memset(chunk.sdf5, -128, sizeof chunk.sdf5);
		chunk.has_surface = false;
		return;
	}

	if (chunk_position.y < -8) {
		memset(chunk.sdf0, 127, sizeof chunk.sdf0);
		memset(chunk.sdf1, 127, sizeof chunk.sdf1);
		memset(chunk.sdf2, 127, sizeof chunk.sdf2);
		memset(chunk.sdf3, 127, sizeof chunk.sdf3);
		memset(chunk.sdf4, 127, sizeof chunk.sdf4);
		memset(chunk.sdf5, 127, sizeof chunk.sdf5);
		chunk.has_surface = false;
		return;
	}
#endif

	chunk->has_surface = true;

	f32 sdf_max_abs = -1;
	f32 sdf_min = max_value<f32>;
	f32 sdf_max = min_value<f32>;
	f32 tmp[CHUNKW][CHUNKW][CHUNKW];

	for (s32 x = 0; x < CHUNKW; ++x) {
	for (s32 y = 0; y < CHUNKW; ++y) {
#if 1
	for (s32 z = 0; z < CHUNKW; z += 8) {
		auto p = chunk_position * CHUNKW + v3s{x,y,z};

		auto d = sdf_func_x8(p);

		f32x8_store(&tmp[x][y][z], d);

		sdf_max = max(sdf_max, d.m256_f32);
		sdf_min = min(sdf_min, d.m256_f32);
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
		chunk.cold->sdf[x][y][z] = s;
#endif
	}
	}
	}

#if PERFECT_SDF
#error not implemented
	// Compute sdf factor that will preserve the most amount of detail
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
		chunk->has_surface = false;
	}
	for (s32 x = 0; x < CHUNKW; ++x) {
	for (s32 y = 0; y < CHUNKW; ++y) {
	for (s32 z = 0; z < CHUNKW; ++z) {
		chunk.cold->sdf0[x][y][z] = (s8)map_clamped(tmp[x][y][z], -sdf_max_abs, sdf_max_abs, -128.f, 127.f);
	}
	}
	}
#else
	if (sdf_max < 0) {
		chunk->average_sdf = -128;
	} else if (sdf_min > 0) {
		chunk->average_sdf = 127;
	} else {
		allocate_sdf(chunk);
		for (s32 x = 0; x < CHUNKW; ++x) {
		for (s32 y = 0; y < CHUNKW; ++y) {
		for (s32 z = 0; z < CHUNKW; ++z) {
			chunk->sdf->_0[x][y][z] = (s8)map_clamped<f32>(tmp[x][y][z], -1, 1, -128.f, 127.999f);
		}
		}
		}
		filter_sdf(chunk);
	}
#endif
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

void generate_grass(Chunk *chunk, v3s chunk_position) {

	if (!chunk->sdf)
		return;

	struct Vertex {
		v3f origin;
		v3f position;
		v3f normal;
		v2f uv;
	};
	StaticList<Vertex, 1024*1024> vertices;

	xorshift32 rng{ max(1,get_hash(chunk_position)) };

	auto put_grass = [&] (v3f position) {
		auto normal = gradient_at(ChunkRelativePosition{.chunk=chunk_position, .local=min(position, V3f(CHUNKW-1.1f))}).value_or({0,1,0});

		f32 angle = next_f32(rng);

		position += v3f {
			next_f32(rng) - 0.5f,
			0,
			next_f32(rng) - 0.5f,
		};

		auto m = m2::rotation(angle * tau);

		f32 u = next_f32(rng) > 0.5f ? 0.5f : 0.0f;
		f32 h = map<f32>(next_f32(rng), 0, 1, 1.5, 2.5);

		for (u32 i = 0; i < 3; ++i) {
			Vertex _0 = {.origin = position, .position = position + V3f(m * v2f{-1,0}, -0.5f).xzy(), .normal = normal, .uv = {u+0.0f,1}};
			Vertex _1 = {.origin = position, .position = position + V3f(m * v2f{-1,0},h-0.5f).xzy(), .normal = normal, .uv = {u+0.0f,0}};
			Vertex _2 = {.origin = position, .position = position + V3f(m * v2f{+1,0}, -0.5f).xzy(), .normal = normal, .uv = {u+0.5f,1}};
			Vertex _3 = {.origin = position, .position = position + V3f(m * v2f{+1,0},h-0.5f).xzy(), .normal = normal, .uv = {u+0.5f,0}};
			vertices.add(_0);
			vertices.add(_1);
			vertices.add(_2);
			vertices.add(_2);
			vertices.add(_1);
			vertices.add(_3);
			m *= m2::rotation(tau / 3);
		}
	};

	for (s32 x = 0; x < CHUNKW; ++x) {
	for (s32 z = 0; z < CHUNKW; ++z) {
		s8 prev = chunk->sdf->_0[x][CHUNKW-1][z];
		for (s32 y = CHUNKW-2; y >= 0; --y) {
			auto cur = chunk->sdf->_0[x][y][z];
			if (cur > 0) {
				if (prev <= 0) {
					put_grass((v3f)v3s{x, y, z} + v3f{0,(f32)cur / (cur - prev),0});
				}
			}
			prev = cur;
		}
	}
	}


	chunk->grass_vb.vertex_count = vertices.count;

	if (vertices.count)
		chunk->grass_vb.view = create_structured_buffer(vertices.span());
}

void generate_trees(Chunk *chunk, v3s chunk_position) {

	if (!chunk->sdf)
		return;

	xorshift32 rng{ max(1,get_hash(chunk_position)) };

	StaticList<TreeInstance, CHUNKW*CHUNKW> trees;

	auto put_tree = [&] (v3f position) {
		if (next_f32(rng) < 0.02f) {
			trees.add({
				.matrix = m3::rotation_r_y(next_f32(rng)*tau) * m3::scale(map<f32>(next_f32(rng), 0, 1, 0.75, 1.25)),
				.position = position,
			});
		}
	};

	for (s32 x = 0; x < CHUNKW; ++x) {
	for (s32 z = 0; z < CHUNKW; ++z) {
		s8 prev = chunk->sdf->_0[x][CHUNKW-1][z];
		for (s32 y = CHUNKW-2; y >= 0; --y) {
			auto cur = chunk->sdf->_0[x][y][z];
			if (cur > 0) {
				if (prev <= 0) {
					put_tree((v3f)v3s{x, y, z} + v3f{0,(f32)cur / (cur - prev),0});
					break;
				}
			}
			prev = cur;
		}
	}
	}

	if (trees.count)
		chunk->trees_instances_buffer.update(trees.span());
}

template <umm dim>
void sdf_to_triangles(v3s lbounds, auto &&sdf, auto &&modify_point, auto &&add_vertex) {

	struct Edge {
		u8 a, b;
	};

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

	v3f corners[8] {
		{0, 0, 0},
		{0, 0, 1},
		{0, 1, 0},
		{0, 1, 1},
		{1, 0, 0},
		{1, 0, 1},
		{1, 1, 0},
		{1, 1, 1},
	};

	ChunkVertex points[dim+3][dim+3][dim+3];

	{
		timed_block(profiler, profile_frame, "point generation");
		for (s32 lx = 0; lx < lbounds.x-1; ++lx) {
		for (s32 ly = 0; ly < lbounds.y-1; ++ly) {
		for (s32 lz = 0; lz < lbounds.z-1; ++lz) {
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
				v3f point = {};
				f32 divisor = 0;

				for (auto &edge : edges) {
					auto a = edge[0];
					auto b = edge[1];
					if ((d[a] > 0) != (d[b] > 0)) {
						point += lerp(corners[a], corners[b], V3f((f32)d[a] / (d[a] - d[b])));
						divisor += 1;
					}
				}
				point /= divisor;
				points[lx][ly][lz].position = modify_point(point + V3f(lx,ly,lz));

				// NOTE: Maybe there is no need to convert this to float and back to int?
				points[lx][ly][lz].normal = encode_normal(normalize((v3f)v3s{
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
				}));
			}
		}
		}
		}
	}

	{
		timed_block(profiler, profile_frame, "triangle generation");
		for (s32 lx = 1; lx < lbounds.x-1; ++lx) {
		for (s32 ly = 1; ly < lbounds.y-1; ++ly) {
		for (s32 lz = 1; lz < lbounds.z-1; ++lz) {
			auto add = [&](v3s v) {
				add_vertex(points[v.x][v.y][v.z]);
			};

			if ((sdf(lx,ly,lz) > 0) != (sdf(lx+1,ly,lz) > 0)) {
				v3s _0 = {lx, ly-1, lz-1};
				v3s _1 = {lx, ly+0, lz-1};
				v3s _2 = {lx, ly-1, lz+0};
				v3s _3 = {lx, ly+0, lz+0};

				if (!(sdf(lx,ly,lz) > 0)) {
					swap(_0, _3);
				}

				add(_0);
				add(_1);
				add(_2);
				add(_1);
				add(_3);
				add(_2);
			}
			if ((sdf(lx,ly,lz) > 0) != (sdf(lx,ly+1,lz) > 0)) {
				v3s _0 = {lx-1, ly, lz-1};
				v3s _1 = {lx+0, ly, lz-1};
				v3s _2 = {lx-1, ly, lz+0};
				v3s _3 = {lx+0, ly, lz+0};

				if (sdf(lx,ly,lz) > 0) {
					swap(_0, _3);
				}

				add(_0);
				add(_1);
				add(_2);
				add(_1);
				add(_3);
				add(_2);
			}
			if ((sdf(lx,ly,lz) > 0) != (sdf(lx,ly,lz+1) > 0)) {
				v3s _0 = {lx-1, ly-1, lz};
				v3s _1 = {lx+0, ly-1, lz};
				v3s _2 = {lx-1, ly+0, lz};
				v3s _3 = {lx+0, ly+0, lz};

				if (!(sdf(lx,ly,lz) > 0)) {
					swap(_0, _3);
				}

				add(_0);
				add(_1);
				add(_2);
				add(_1);
				add(_3);
				add(_2);
			}
		}
		}
		}
	}
}

using ChunkVertexArena = StaticList<ChunkVertex, 1024*1024>;

template <u32 lodw>
void generate_chunk_lod(Chunk *chunk, v3s chunk_position, ChunkVertexArena &vertices) {
	v3s lbounds = V3s(lodw);
	if constexpr (lodw <= 2) {
		if (chunk->neighbor_mask.x) lbounds.x += 2;
		if (chunk->neighbor_mask.y) lbounds.y += 2;
		if (chunk->neighbor_mask.z) lbounds.z += 2;
	} else {
		if (chunk->neighbor_mask.x) lbounds.x += 4;
		if (chunk->neighbor_mask.y) lbounds.y += 4;
		if (chunk->neighbor_mask.z) lbounds.z += 4;
	}

	auto get_sdf = [&] (Chunk *chunk) {
		     if constexpr (lodw == 32) return chunk->sdf->_0;
		else if constexpr (lodw == 16) return chunk->sdf->_1;
		else if constexpr (lodw ==  8) return chunk->sdf->_2;
		else if constexpr (lodw ==  4) return chunk->sdf->_3;
		else if constexpr (lodw ==  2) return chunk->sdf->_4;
		else if constexpr (lodw ==  1) return chunk->sdf->_5;
		else static_assert(false);
	};

	Chunk *neighbors[8];
	neighbors[0b000] = chunk;
	neighbors[0b001] = get_chunk(chunk_position+v3s{0,0,1});
	neighbors[0b010] = get_chunk(chunk_position+v3s{0,1,0});
	neighbors[0b011] = get_chunk(chunk_position+v3s{0,1,1});
	neighbors[0b100] = get_chunk(chunk_position+v3s{1,0,0});
	neighbors[0b101] = get_chunk(chunk_position+v3s{1,0,1});
	neighbors[0b110] = get_chunk(chunk_position+v3s{1,1,0});
	neighbors[0b111] = get_chunk(chunk_position+v3s{1,1,1});

	auto sdf = [&](s32 x, s32 y, s32 z) -> s8 {
		x -= (x == lodw*2);
		y -= (y == lodw*2);
		z -= (z == lodw*2);

		auto lx = x - (x >= lodw) * lodw;
		auto ly = y - (y >= lodw) * lodw;
		auto lz = z - (z >= lodw) * lodw;

		u32 neighbor_index =
			((x >= lodw) << 2) |
			((y >= lodw) << 1) |
			((z >= lodw) << 0);

		auto neighbor = neighbors[neighbor_index];

		if (neighbor->sdf)
			return get_sdf(neighbor)[lx][ly][lz];

		return neighbor->average_sdf;
	};

	u32 start_vertex = vertices.count;

	sdf_to_triangles<lodw>(lbounds, sdf,
		[&](v3f point) {
			point *= CHUNKW/lodw;
			point += 0.5f * (1 << (log2(CHUNKW) - log2(lodw))) - 0.5f;
			return point;
		},
		[&](ChunkVertex vertex) {
			vertices.add(vertex);
		}
	);

	auto lod_index = log2(CHUNKW) - log2(lodw);

	chunk->sdf_vb_vertex_offset[lod_index] = start_vertex;
	chunk->sdf_vb_vertex_count [lod_index] = vertices.count - start_vertex;

	if (lod_index == 0) {
		for (u32 i = start_vertex; i != vertices.count; ++i)
			chunk->sdf_vertex_positions.add(vertices[i].position);
	}

	chunk->frames_since_remesh = 0;
}

void update_chunk_mesh(Chunk *chunk, v3s chunk_position) {
	timed_function(profiler, profile_frame);

#if DEBUG_CHUNK_THREAD_ACCESS
	u32 locked_by;
	if (!try_lock(chunk->mutex, &locked_by)) {
		with(ConsoleColor::red, print("Attempt to access the same chunk from different threads: {} and {}\n", locked_by, get_current_thread_id()));
		invalid_code_path("shared chunk access must not happen!");
	}
	defer { unlock(chunk->mutex); };
#endif

	ChunkVertexArena vertices;
	generate_chunk_lod<1 >(chunk, chunk_position, vertices);
	generate_chunk_lod<2 >(chunk, chunk_position, vertices);
	generate_chunk_lod<4 >(chunk, chunk_position, vertices);
	generate_chunk_lod<8 >(chunk, chunk_position, vertices);
	generate_chunk_lod<16>(chunk, chunk_position, vertices);
	generate_chunk_lod<32>(chunk, chunk_position, vertices);


	if (vertices.count) {
		chunk->sdf_vb = create_structured_buffer(vertices.span());
	}

	chunk->sdf_mesh_generated = true;
}

void start() {

}

FrustumPlanes frustum;

bool chunk_in_view(v3s position) {
	auto relative_position = (v3f)((position-camera->position.chunk)*CHUNKW);
	return contains_sphere(frustum, relative_position+V3f(CHUNKW/2), sqrt3*CHUNKW);
}

extern "C" const Array<v3s8, pow3(DRAWD*2+1)> grid_map;

u32 thread_count;

u32 first_not_generated_sdf_chunk_index = 0;
u32 first_not_fully_meshed_chunk_index = 0;
u32 fully_meshed_chunk_index_end = 0;

s32 n_sdfs_can_generate_this_frame;
u32 remesh_count = 0;

s32 get_chunk_lod_index(v3s p) {
	//auto distance = max(absolute(p - camera->position.chunk));
	auto distance = length((v3f)(p - camera->position.chunk) * CHUNKW - camera->position.local) / CHUNKW;
	return log2((s32)max(distance, 1));
}

void remesh_blocks(Chunk *chunk) {
	assert(chunk->blocks);

	StaticList<ChunkVertex, 1024*1024> vertices;

	for (u32 x = 0; x < CHUNKW; ++x) {
	for (u32 y = 0; y < CHUNKW; ++y) {
	for (u32 z = 0; z < CHUNKW; ++z) {
		if ((*chunk->blocks)[x][y][z].solid) {
			auto _000 = (v3f)v3u{x,y,z} + v3f{0,0,0};
			auto _001 = (v3f)v3u{x,y,z} + v3f{0,0,1};
			auto _010 = (v3f)v3u{x,y,z} + v3f{0,1,0};
			auto _011 = (v3f)v3u{x,y,z} + v3f{0,1,1};
			auto _100 = (v3f)v3u{x,y,z} + v3f{1,0,0};
			auto _101 = (v3f)v3u{x,y,z} + v3f{1,0,1};
			auto _110 = (v3f)v3u{x,y,z} + v3f{1,1,0};
			auto _111 = (v3f)v3u{x,y,z} + v3f{1,1,1};

			// front
			vertices.add({.position=_011,.normal=encode_normal({0,0,1})});
			vertices.add({.position=_001,.normal=encode_normal({0,0,1})});
			vertices.add({.position=_101,.normal=encode_normal({0,0,1})});
			vertices.add({.position=_111,.normal=encode_normal({0,0,1})});
			vertices.add({.position=_011,.normal=encode_normal({0,0,1})});
			vertices.add({.position=_101,.normal=encode_normal({0,0,1})});

			// back
			vertices.add({.position=_000,.normal=encode_normal({0,0,-1})});
			vertices.add({.position=_010,.normal=encode_normal({0,0,-1})});
			vertices.add({.position=_100,.normal=encode_normal({0,0,-1})});
			vertices.add({.position=_010,.normal=encode_normal({0,0,-1})});
			vertices.add({.position=_110,.normal=encode_normal({0,0,-1})});
			vertices.add({.position=_100,.normal=encode_normal({0,0,-1})});

			// right
			vertices.add({.position=_101,.normal=encode_normal({1,0,0})});
			vertices.add({.position=_100,.normal=encode_normal({1,0,0})});
			vertices.add({.position=_110,.normal=encode_normal({1,0,0})});
			vertices.add({.position=_111,.normal=encode_normal({1,0,0})});
			vertices.add({.position=_101,.normal=encode_normal({1,0,0})});
			vertices.add({.position=_110,.normal=encode_normal({1,0,0})});

			// left
			vertices.add({.position=_000,.normal=encode_normal({-1,0,0})});
			vertices.add({.position=_001,.normal=encode_normal({-1,0,0})});
			vertices.add({.position=_010,.normal=encode_normal({-1,0,0})});
			vertices.add({.position=_001,.normal=encode_normal({-1,0,0})});
			vertices.add({.position=_011,.normal=encode_normal({-1,0,0})});
			vertices.add({.position=_010,.normal=encode_normal({-1,0,0})});

			// top
			vertices.add({.position=_010,.normal=encode_normal({0,1,0})});
			vertices.add({.position=_011,.normal=encode_normal({0,1,0})});
			vertices.add({.position=_110,.normal=encode_normal({0,1,0})});
			vertices.add({.position=_011,.normal=encode_normal({0,1,0})});
			vertices.add({.position=_111,.normal=encode_normal({0,1,0})});
			vertices.add({.position=_110,.normal=encode_normal({0,1,0})});

			// bottom
			vertices.add({.position=_001,.normal=encode_normal({0,-1,0})});
			vertices.add({.position=_000,.normal=encode_normal({0,-1,0})});
			vertices.add({.position=_100,.normal=encode_normal({0,-1,0})});
			vertices.add({.position=_101,.normal=encode_normal({0,-1,0})});
			vertices.add({.position=_001,.normal=encode_normal({0,-1,0})});
			vertices.add({.position=_100,.normal=encode_normal({0,-1,0})});
		}
	}
	}
	}

	chunk->blocks_vb.vertex_count = vertices.count;
	if (vertices.count) {
		chunk->blocks_vb.view = create_structured_buffer(vertices.span());
	}

	update_chunk_mask(chunk, get_chunk_position(chunk));
}

void generate_chunks_around() {
	auto generate_timer = create_precise_timer();
	defer { generate_time.set(get_time(generate_timer)); };

	timed_function(profiler, profile_frame);

	s32 n_sdfs_generated = 0;

	if (sdfs_generated_per_frame) {
		auto avg = (f32)generate_time.current * thread_count / sdfs_generated_per_frame;
		if (smoothed_average_generation_time == 0) {
			smoothed_average_generation_time = avg;
		} else {
			if (avg > smoothed_average_generation_time)
				smoothed_average_generation_time = avg;
			else
				smoothed_average_generation_time = lerp(smoothed_average_generation_time, avg, target_frame_time);
		}
		sdfs_generated_per_frame = 0;
	}

	// n_sdfs_can_generate_this_frame = 4;

	f32 update_time_except_gen_time = frame_time - generate_time.current;
	assert(update_time_except_gen_time >= 0);

	f32 available_time = max(0, target_frame_time - update_time_except_gen_time);

	n_sdfs_can_generate_this_frame =
		smoothed_average_generation_time == 0 ?
		thread_count :
		chunk_generation_amount_factor * thread_count * max(1, floor_to_int(available_time / smoothed_average_generation_time));

	auto work = make_work_queue(thread_pool);

	if (any_true(camera_chunk_last_frame != camera->position.chunk)) {
		timed_block(profiler, profile_frame, "remove chunks");
		first_not_generated_sdf_chunk_index = 0;
		first_not_fully_meshed_chunk_index = 0;

		for (s32 x = camera_chunk_last_frame.x-DRAWD; x <= camera_chunk_last_frame.x+DRAWD; ++x) {
		for (s32 y = camera_chunk_last_frame.y-DRAWD; y <= camera_chunk_last_frame.y+DRAWD; ++y) {
		for (s32 z = camera_chunk_last_frame.z-DRAWD; z <= camera_chunk_last_frame.z+DRAWD; ++z) {
			if (camera->position.chunk.x-DRAWD <= x && x <= camera->position.chunk.x+DRAWD)
			if (camera->position.chunk.y-DRAWD <= y && y <= camera->position.chunk.y+DRAWD)
			if (camera->position.chunk.z-DRAWD <= z && z <= camera->position.chunk.z+DRAWD)
				continue;

			auto chunk_position = v3s{x,y,z};

			auto r = absolute(chunk_position - camera->position.chunk);
			assert(r.x > DRAWD || r.y > DRAWD || r.z > DRAWD);
			auto chunk = get_chunk(chunk_position);

			if (chunk->sdf_vb) {
				chunk->sdf_vb->Release();
				chunk->sdf_vb = 0;
			}
			memset(chunk->sdf_vb_vertex_count, 0, sizeof(chunk->sdf_vb_vertex_count));
			memset(chunk->sdf_vb_vertex_offset, 0, sizeof(chunk->sdf_vb_vertex_offset));

			if (chunk->grass_vb.view) {
				chunk->grass_vb.view->Release();
				chunk->grass_vb.view = 0;
			}
			chunk->grass_vb.vertex_count = 0;

			if (chunk->blocks_vb.view) {
				chunk->blocks_vb.view->Release();
				chunk->blocks_vb.view = 0;
			}
			chunk->blocks_vb.vertex_count = 0;

#if USE_INDICES
			if (chunk->index_buffer) {
				chunk->index_buffer->Release();
				chunk->index_buffer = 0;
			}
			chunk->indices_count = 0;
#endif
			chunk->has_surface = false;
			chunk->sdf_generated = false;
			chunk->neighbor_mask = {};

			free(chunk->trees_instances_buffer);

			if (chunk->needs_saving) {
				chunk->needs_saving = false;
				save_chunk(chunk, chunk_position);
			}

			if (chunk->sdf) {
				free_sdf(chunk);
			}

			if (chunk->blocks) {
				free_blocks(chunk);
			}

			free(chunk->sdf_vertex_positions);

			chunk->lod_t = 1;
		}
		}
		}

		// RECREATE FARLANDS
		{
			if (farlands_vb.view) {
				farlands_vb.view->Release();
				farlands_vb.view = 0;
			}

			StaticList<ChunkVertex, 1024*1024> vertices;

			s32 const dim = FARD*2 + 1;

			s8 sdf[dim+3][dim+3][dim+3];
			for (s32 x = 0; x < dim; ++x) {
			for (s32 y = 0; y < dim; ++y) {
			for (s32 z = 0; z < dim; z += 8) {
				auto d = f32x8_clamp(sdf_func_x8((v3s{x,y,z} + camera->position.chunk - dim/2)*CHUNKW, CHUNKW), f32x8_set1(-128), f32x8_set1(127));
				for (u32 i = 0; i < 8; ++i)
					sdf[x][y][z+i] = (s8)d.m256_f32[i];
			}
			}
			}

			sdf_to_triangles<dim>(V3s(dim),
				[&](s32 x, s32 y, s32 z) {
					return sdf[x][y][z];
				},
				[&](v3f point) {
					point = (point - dim/2) * CHUNKW;
					//point += 8;
					return point;
				},
				[&](ChunkVertex vertex) {
					vertices.add(vertex);
				}
			);

			farlands_vb.vertex_count = vertices.count;
			if (vertices.count) {
				farlands_vb.view = create_structured_buffer(vertices.span());
			}
		}
	}

	{
		timed_block(profiler, profile_frame, "generate sdfs");
		remesh_count = 0;
		bool reached_not_generated_sdf = false;
		bool reached_not_generated_mesh = false;

		for (
			auto pp_small = grid_map.begin() + min(first_not_generated_sdf_chunk_index, first_not_fully_meshed_chunk_index);
			pp_small != grid_map.end();
			++pp_small
		) {
			auto &p_small = *pp_small;
			auto p = (v3s)p_small;
			auto p_index = pp_small - grid_map.data;
			auto chunk_position = camera->position.chunk + p;
			auto chunk = get_chunk(chunk_position);
			if (chunk->sdf_generated) {
				if (chunk->needs_filter_and_remesh) {
					// always update requested remeshes
					chunk->needs_filter_and_remesh = false;
					work.push([chunk = chunk, chunk_position] {
						filter_sdf(chunk);
						update_chunk_mesh(chunk, chunk_position);
						update_chunk_mask(chunk, chunk_position);
					});
					remesh_count++;
				} else {
					if (remesh_count < n_sdfs_can_generate_this_frame) {
						auto new_neighbor_mask = get_neighbor_mask(chunk_position);
						auto r = chunk_position == camera->position.chunk + DRAWD;
						if (chunk->neighbor_mask != new_neighbor_mask) {
							if ((new_neighbor_mask.x || r.x) && (new_neighbor_mask.y || r.y) && (new_neighbor_mask.z || r.z)) {
								chunk->neighbor_mask = new_neighbor_mask;
								work.push([chunk, chunk_position] {
									update_chunk_mesh(chunk, chunk_position);
									update_chunk_mask(chunk, chunk_position);
								});
								remesh_count++;
							}
						}
					}
				}
			} else {
				if (!reached_not_generated_sdf) {
					reached_not_generated_sdf = true;
					first_not_generated_sdf_chunk_index = p_index;
				}

				if (n_sdfs_generated < n_sdfs_can_generate_this_frame) {
					n_sdfs_generated += 1;

					chunk->lod_previous_frame = get_chunk_lod_index(chunk_position);

					if (auto found = saved_chunks.find(chunk_position)) {
						work.push([chunk = chunk, chunk_position, found]{
							timed_block(profiler, profile_frame, "load saved chunk");
							allocate_sdf(chunk);
							memcpy(chunk->sdf->_0, found->value.sdf, sizeof(found->value.sdf));
							filter_sdf(chunk);
							chunk->sdf_generated = true;
							chunk->has_surface = true;

							allocate_blocks(chunk);
							memcpy(chunk->blocks, found->value.blocks, sizeof(found->value.blocks));
							remesh_blocks(chunk);

							generate_trees(chunk, chunk_position);

							generate_grass(chunk, chunk_position);
							update_chunk_mask(chunk, chunk_position);
						});
					} else {
						work.push([chunk = chunk, chunk_position]{
							generate_sdf(chunk, chunk_position);
							if (chunk->has_surface) {
								generate_grass(chunk, chunk_position);
								generate_trees(chunk, chunk_position);
							}
							update_chunk_mask(chunk, chunk_position);
						});
					}
				}
			}

			bool has_full_mesh = false;
			if (chunk->sdf_generated) {
				auto e = chunk_position == camera->position.chunk + DRAWD;
				// Chunks that are not on the edge must have all neighbors.
				// Mesh of the chunk on the edge of draw distance in positive X/Y/Z direction(s)
				// will be considered full without the neighbor in that direction(s).
				has_full_mesh =
					(chunk->neighbor_mask.x | e.x) &
					(chunk->neighbor_mask.y | e.y) &
					(chunk->neighbor_mask.z | e.z);
			}

			if (has_full_mesh) {
				fully_meshed_chunk_index_end = p_index+1;
			} else {
				if (!reached_not_generated_mesh) {
					reached_not_generated_mesh = true;
					first_not_fully_meshed_chunk_index = p_index;
				}
			}
		}
	}
	{
#if 0
		timed_block(profiler, profile_frame, "remesh chunks");


		remesh_count = 0;

		bool reached_not_generated = false;

		for (auto pp_small = grid_map.begin() + first_not_fully_meshed_chunk_index; pp_small != grid_map.begin() + iter_from_center_offset_sdf_end; ++pp_small) {
			auto &p_small = *pp_small;
			auto p = (v3s)p_small;
			auto chunk_position = camera->position.chunk + p;

			auto lod_index = get_chunk_lod_index(p);

			//if (visible)
			//	if (lodw == 32)
			//		debug_break();

			bool did_remesh = false;

			auto &chunk = get_chunk(chunk_position);

			if (chunk.sdf_generated) {
			}

			bool has_full_mesh =
				any_true(chunk_position == camera->position.chunk + DRAWD) ?
				chunk.sdf_generated && (chunk.neighbor_mask.x || chunk.neighbor_mask.y || chunk.neighbor_mask.z) : // these are on the edge of draw distance, they will never have neighbors therefore full mesh
				chunk.sdf_generated && chunk.neighbor_mask.x && chunk.neighbor_mask.y && chunk.neighbor_mask.z; //

			if (!has_full_mesh) {
				if (!reached_not_generated) {
					reached_not_generated = true;
					first_not_fully_meshed_chunk_index = pp_small - grid_map.data;
				}
			}
		}
		_end2:
		//print("remesh_count: {}\n", remesh_count);
#endif
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

	origin_crp.ensure_valid();

	RayHit result = {};

	if (length(direction) < 0.000001f)
		return {};
	direction = normalize(direction);

	auto end   = origin_crp + direction * max_distance;

	v3s _min = min(origin_crp.chunk, end.chunk) - 1; // account for extended mesh from chunks behind
	v3s _max = max(origin_crp.chunk, end.chunk);

	for (s32 x = _min.x; x <= _max.x; ++x) {
	for (s32 y = _min.y; y <= _max.y; ++y) {
	for (s32 z = _min.z; z <= _max.z; ++z) {
		v3s chunk_position = {x,y,z};
		auto chunk = get_chunk(chunk_position);

		auto origin = origin_crp;
		origin.chunk -= chunk_position;

		auto ray = ray_origin_direction(origin.to_v3f(), direction);

		if (raycast(ray, aabb_min_max(v3f{}, V3f(CHUNKW+2)), true)) {
#if USE_INDICES
			for (u32 i = 0; i < chunk.indices.count; i += 3) {
				auto a = chunk.vertices[chunk.indices[i+0]];
				auto b = chunk.vertices[chunk.indices[i+1]];
				auto c = chunk.vertices[chunk.indices[i+2]];
#else
			for (u32 i = 0; i < chunk->sdf_vertex_positions.count; i += 3) {
				auto a = chunk->sdf_vertex_positions[i+0];
				auto b = chunk->sdf_vertex_positions[i+1];
				auto c = chunk->sdf_vertex_positions[i+2];
#endif

				if (auto hit = raycast(ray, triangle{a,b,c})) {
					if (!result || hit.distance < result.distance) {
						result = RayHit{
							.did_hit = true,
							.position = {
								.chunk = chunk_position,
								.local = hit.position
							},
							.chunk = chunk,
							.distance = hit.distance,
							.normal = hit.normal,
						};
					}
				}
			}

			if (chunk->blocks) {
				for (s32 x = 0; x < CHUNKW; ++x) {
				for (s32 y = 0; y < CHUNKW; ++y) {
				for (s32 z = 0; z < CHUNKW; ++z) {
					if ((*chunk->blocks)[x][y][z].solid) {
						auto box = aabb_min_size((v3f)v3s{x,y,z}, V3f(1));

						if (auto hit = raycast(ray, box)) {

							if (!result || hit.distance < result.distance) {
								result = RayHit{
									.did_hit = true,
									.position = {
										.chunk = chunk_position,
										.local = hit.position
									},
									.chunk = chunk,
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
		}
	}
	}
	}

	return result;
}

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

s32 brush_size = 8;

void mark_chunk_sdf_modified(Chunk *chunk) {
	chunk->needs_saving = true;
	chunk->needs_filter_and_remesh = true;
	first_not_generated_sdf_chunk_index = first_not_fully_meshed_chunk_index = 0;
}

// TODO: room for optimization is very big
void add_sdf_sphere(ChunkRelativePosition position, s32 radius, s32 strength) {
	timed_function(profiler, profile_frame);

	v3s center = round_to_int(position.local);

	v3s cmin = position.chunk + floor(center - radius-4, V3s(CHUNKW)) / CHUNKW;
	v3s cmax = position.chunk + ceil (center + radius+2, V3s(CHUNKW)) / CHUNKW;

	auto add_sdf = [&](s32 x, s32 y, s32 z, s32 delta) {
		if (delta == 0)
			return;

		v3s c = position.chunk;

		v3s f = {
			floor(x, CHUNKW),
			floor(y, CHUNKW),
			floor(z, CHUNKW),
		};

		c.x += f.x / CHUNKW;
		c.y += f.y / CHUNKW;
		c.z += f.z / CHUNKW;

		x -= f.x;
		y -= f.y;
		z -= f.z;

		auto chunk = get_chunk(c);

		if (!chunk->sdf_generated)
			return;
		chunk->sdf->_0[x][y][z] = clamp(chunk->sdf->_0[x][y][z] + delta, -128, 127);
	};

	for (s32 x = cmin.x; x < cmax.x; ++x) {
	for (s32 y = cmin.y; y < cmax.y; ++y) {
	for (s32 z = cmin.z; z < cmax.z; ++z) {
		auto chunk = get_chunk(x,y,z);
		if (chunk->sdf_generated) {
			if (!chunk->sdf) {
				allocate_sdf(chunk);
				// NOTE: memset all lods.
				memset(chunk->sdf, chunk->average_sdf, sizeof(*chunk->sdf));
			}
		}
	}
	}
	}
	for (s32 x = -radius; x <= radius; ++x) {
	for (s32 y = -radius; y <= radius; ++y) {
	for (s32 z = -radius; z <= radius; ++z) {
		auto c = center + v3s{x,y,z};
		s32 d = (1 - min(radius, length(V3f(x,y,z))) / radius) * strength;
		add_sdf(c.x, c.y, c.z, d);
	}
	}
	}

	for (s32 x = cmin.x; x < cmax.x; ++x) {
	for (s32 y = cmin.y; y < cmax.y; ++y) {
	for (s32 z = cmin.z; z < cmax.z; ++z) {
		auto chunk = get_chunk(x,y,z);
		if (chunk->sdf_generated) {
			mark_chunk_sdf_modified(chunk);
		}
	}
	}
	}
}

Optional<s8> sdf_at(v3s chunk_position, v3s l) {
	auto chunk = get_chunk(chunk_position);
	if (!chunk->sdf_generated)
		return {};
	if (chunk->sdf)
		return chunk->sdf->_0[l.x][l.y][l.z];
	return chunk->average_sdf;
}
Optional<s8> sdf_at(ChunkRelativePosition position) {
	return sdf_at(position.chunk, floor_to_int(position.local));
}

Optional<f32> sdf_at_interpolated(ChunkRelativePosition position) {
	auto chunk = get_chunk(position.chunk);
	if (!chunk->sdf_generated)
		return {};

	auto l = floor_to_int(position.local);
	auto t = position.local - (v3f)l;

	f32 d[8];

	if (all_true(l+1 < CHUNKW)) {
		if (chunk->sdf) {
			d[0b000] = chunk->sdf->_0[l.x+0][l.y+0][l.z+0];
			d[0b001] = chunk->sdf->_0[l.x+0][l.y+0][l.z+1];
			d[0b010] = chunk->sdf->_0[l.x+0][l.y+1][l.z+0];
			d[0b011] = chunk->sdf->_0[l.x+0][l.y+1][l.z+1];
			d[0b100] = chunk->sdf->_0[l.x+1][l.y+0][l.z+0];
			d[0b101] = chunk->sdf->_0[l.x+1][l.y+0][l.z+1];
			d[0b110] = chunk->sdf->_0[l.x+1][l.y+1][l.z+0];
			d[0b111] = chunk->sdf->_0[l.x+1][l.y+1][l.z+1];
		} else {
			d[0b000] =
			d[0b001] =
			d[0b010] =
			d[0b011] =
			d[0b100] =
			d[0b101] =
			d[0b110] =
			d[0b111] = chunk->average_sdf;
		}
	} else {
		bool fail = false;
		auto get = [&] (v3f offset){
			auto s = sdf_at(position + offset);
			if (s)
				return s.value_unchecked();
			fail = true;
			return (s8)0;
		};
		d[0b000] = get({0,0,0});
		d[0b001] = get({0,0,1});
		d[0b010] = get({0,1,0});
		d[0b011] = get({0,1,1});
		d[0b100] = get({1,0,0});
		d[0b101] = get({1,0,1});
		d[0b110] = get({1,1,0});
		d[0b111] = get({1,1,1});
		if (fail)
			return {};
	}
	return
		lerp(lerp(lerp(d[0b000],
		               d[0b001], t.z),
		          lerp(d[0b010],
		               d[0b011], t.z), t.y),
		     lerp(lerp(d[0b100],
		               d[0b101], t.z),
		          lerp(d[0b110],
		               d[0b111], t.z), t.y), t.x);
}

Optional<v3f> gradient_at(ChunkRelativePosition position) {
	auto chunk = get_chunk(position.chunk);
	if (!chunk->sdf_generated)
		return {};

	auto l = floor_to_int(position.local);

	s8 d[8];

	if (all_true(l+1 < CHUNKW)) {
		if (chunk->sdf) {
			d[0b000] = chunk->sdf->_0[l.x+0][l.y+0][l.z+0];
			d[0b001] = chunk->sdf->_0[l.x+0][l.y+0][l.z+1];
			d[0b010] = chunk->sdf->_0[l.x+0][l.y+1][l.z+0];
			d[0b011] = chunk->sdf->_0[l.x+0][l.y+1][l.z+1];
			d[0b100] = chunk->sdf->_0[l.x+1][l.y+0][l.z+0];
			d[0b101] = chunk->sdf->_0[l.x+1][l.y+0][l.z+1];
			d[0b110] = chunk->sdf->_0[l.x+1][l.y+1][l.z+0];
			d[0b111] = chunk->sdf->_0[l.x+1][l.y+1][l.z+1];
		} else {
			d[0b000] =
			d[0b001] =
			d[0b010] =
			d[0b011] =
			d[0b100] =
			d[0b101] =
			d[0b110] =
			d[0b111] = chunk->average_sdf;
		}
	} else {
		bool fail = false;
		auto get = [&](v3f offset){auto s=sdf_at(position + offset);if(s)return s.value_unchecked();fail=true;return(s8)0;};
		d[0b000] = get({0,0,0});
		d[0b001] = get({0,0,1});
		d[0b010] = get({0,1,0});
		d[0b011] = get({0,1,1});
		d[0b100] = get({1,0,0});
		d[0b101] = get({1,0,1});
		d[0b110] = get({1,1,0});
		d[0b111] = get({1,1,1});
		if (fail)
			return {};
	}

	auto v = (v3f)v3s{
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
	};

	if (length(v) < 0.000001)
		return {};

	return normalize(v);
}

s32 debug_selected_lod = -1;

bool is_server;
bool is_connected;
bool another_thread_is_connected;
bool opponent_received_chunks;
bool world_transfer_ended;

enum class NetMessageKind : u8 {
	brush,
	set_position,
	//set_chunk,
	//end_chunk_transfer,
};

// TODO: separate structs instead of everything in NetMessage,
// Queue<u8> net_queue;

#pragma pack(push, 1)
struct NetMessage {
	NetMessageKind kind;
	union {
		struct {
			ChunkRelativePosition position;
			s32 radius;
			s32 strength;
		} brush;
		ChunkRelativePosition set_position;
		//struct {
		//	v3s position;
		//	s8 sdf[CHUNKW][CHUNKW][CHUNKW];
		//} set_chunk;
		//struct {} end_chunk_transfer;
	};
};
#pragma pack(pop)

Queue<NetMessage> net_queue;
Mutex net_queue_mutex;
net::Socket opponent_socket;

void send_message(NetMessage const &message) {
	if (!is_connected)
		return;

	bool success;
	switch (message.kind) {
		using enum NetMessageKind;
#define C(x) case x: success = send_all(opponent_socket, Span((u8 *)&message, sizeof(message.x) + 1)); break;

		C(brush)
		C(set_position)
		//C(set_chunk)
		//C(end_chunk_transfer)

#undef C
		default: invalid_code_path();
	}
	if (!success) {
		print("send failed\n");
		print_wsa_error();
	}
}

void add_sdf_sphere_synchronized(ChunkRelativePosition position, s32 brush_size, s32 strength) {
	add_sdf_sphere(position, brush_size, strength);
	send_message({
		.kind = NetMessageKind::brush,
		.brush = {
			.position = position,
			.radius = brush_size,
			.strength = strength,
		}
	});
}

void draw_text(Span<utf8> str, u32 size, v2f position, bool ndc_position = false) {
	auto font = get_font_at_size(font_collection, size);
	ensure_all_chars_present(str, font);
	auto placed_chars = place_text(str, font);
	defer { free(placed_chars); };

	if (font_vertex_buffer) {
		font_vertex_buffer->Release();
		font_vertex_buffer = 0;
	}
	{
		StaticList<FontVertex, 1024*1024> vertices;

		for (auto c : placed_chars) {
			FontVertex face[] {
				{{c.position.min.x, c.position.min.y}, {c.uv.min.x, c.uv.min.y}},
				{{c.position.min.x, c.position.max.y}, {c.uv.min.x, c.uv.max.y}},
				{{c.position.max.x, c.position.min.y}, {c.uv.max.x, c.uv.min.y}},
				{{c.position.max.x, c.position.max.y}, {c.uv.max.x, c.uv.max.y}},
			};

			vertices.add(face[0]);
			vertices.add(face[2]);
			vertices.add(face[1]);
			vertices.add(face[1]);
			vertices.add(face[2]);
			vertices.add(face[3]);
		}
		for (auto &v : vertices) {
			if (ndc_position)
				v.position = map(v.position, v2f{}, (v2f)window_client_size, v2f{0, 0}, v2f{2,-2}) + position;
			else
				v.position = map(v.position + position, v2f{}, (v2f)window_client_size, v2f{-1, 1}, v2f{1,-1});
		}

		font_vertex_buffer = create_structured_buffer(vertices.span());

		immediate_context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
		immediate_context->VSSetShaderResources(0, 1, &font_vertex_buffer);
		immediate_context->PSSetShaderResources(0, 1, &font->texture);
		immediate_context->VSSetShader(font_vs, 0, 0);
		immediate_context->PSSetShader(font_ps, 0, 0);
		immediate_context->OMSetBlendState(font_blend, {}, -1);

		immediate_context->Draw(vertices.count, 0);
	}
}

List<utf8> chunk_info_string;
List<utf8> allocation_info_string;
PreciseTimer the_timer;
u32 frame_number;

void update() {
	is_connected = another_thread_is_connected;

	// NOTE: all camera->position modifications must happen AFTER saving position at previous frame.
	camera_chunk_last_frame = camera->position.chunk;

	timed_function(profiler, profile_frame);

	if (key_down(Key_escape)) {
		if (cursor_is_locked)
			unlock_cursor();
		else
			lock_cursor();
	}

	if (key_down('3')) {
		if (debug_selected_lod != -1)
			debug_selected_lod -= 1;
	}
	if (key_down('4')) {
		if (debug_selected_lod != 5)
			debug_selected_lod += 1;
	}
	if (key_down(Key_minus)) {
		if (chunk_generation_amount_factor != 1)
			chunk_generation_amount_factor /= 2;
	}
	if (key_down(Key_plus)) {
		if (chunk_generation_amount_factor != 64)
			chunk_generation_amount_factor *= 2;
	}

	if (key_down(Key_left_bracket )) draw_grass = !draw_grass;
	if (key_down(Key_right_bracket)) draw_trees = !draw_trees;

	v3f camera_position_delta {
		key_held(Key_d) - key_held(Key_a),
		key_held(Key_e) - key_held(Key_q),
		key_held(Key_w) - key_held(Key_s),
	};

	if (key_held(Key_control))
		camera_fov = clamp(camera_fov - mouse_wheel_delta*5, 5.f, 175.f);
	else {
		if (mouse_wheel_delta >= 0)
			brush_size <<= (u32)mouse_wheel_delta;
		else
			brush_size >>= (u32)-mouse_wheel_delta;
		brush_size = clamp(brush_size, 1, CHUNKW);
	}

	if (cursor_is_locked) {
		target_camera_angles.y += mouse_delta.x * camera_fov * 0.00002f;
		target_camera_angles.x += mouse_delta.y * camera_fov * 0.00002f;
	}
	camera_angles = lerp(camera_angles, target_camera_angles, V3f(target_frame_time * 20));

	auto camera_rotation = m4::rotation_r_zxy(camera_angles);


	if (key_down('H')) {
		camera->position = camera_prev_position = {0,1,0};
	}

	{
		switch (camera_mode) {
			case CameraMode::fly: {
				camera_prev_position = camera->position;
				camera->position += camera_rotation * camera_position_delta * target_frame_time * 32 * (key_held(Key_shift) ? 10 : 1);

				if (key_down('F')) {
					camera_mode = CameraMode::walk;
					camera_prev_position = camera->position;
				}
				break;
			}
			case CameraMode::walk: {
				camera->position += m4::rotation_r_y(camera_angles.y) * camera_position_delta * v3f{1,0,1} * target_frame_time * (key_held(Key_shift) ? 2 : 1);
				if (key_down(' '))
					camera->position += v3f{0,target_frame_time*7,0};

				//auto velocity = (camera->position - camera_prev_position).to_v3f();

				//f32 const max_sideways_velocity = 8*target_frame_time;
				//if (length(velocity.xz()) >= max_sideways_velocity) {
				//	auto xz = normalize(velocity.xz()) * max_sideways_velocity;
				//	velocity = {
				//		xz.x,
				//		velocity.y,
				//		xz.y,
				//	};
				//}

				f32 camera_height = 1.7f;
				f32 player_height = 1.8f;
				f32 player_radius = .2f;

				v3f velocity_multiplier = length(camera_position_delta.xz()) < 0.001f ? v3f{.0,1,.0} : v3f{.8,1,.8};
				apply_physics(camera->position, camera_prev_position, velocity_multiplier, v3f{0,-9.8f,0}, {
					{-player_radius, -camera_height, -player_radius},
					{+player_radius, player_height-camera_height, +player_radius},
				});

				if (key_down('F')) {
					camera_mode = CameraMode::fly;
				}
				break;
			}
		}
	}

#define e(name) for (auto &component : components<name>) component.update();
ENUMERATE_COMPONENTS
#undef e

#if 0
	if (is_server && is_connected && !opponent_received_chunks) {
		opponent_received_chunks = true;

		for (auto &[position, chunk] : saved_chunks) {
			NetMessage message;
			message.kind = NetMessageKind::set_chunk,
			message.set_chunk.position = position,
			memcpy(message.set_chunk.sdf, chunk.sdf, sizeof(chunk.sdf));
			send_message(message);
			print("sent chunk {}\n", position);
		}
		send_message({.kind = NetMessageKind::end_chunk_transfer});
		print("sent end_chunk_transfer\n");
	}
#endif

	auto rotproj = m4::perspective_left_handed((f32)window_client_size.x / window_client_size.y, radians(camera_fov), 0.1, CHUNKW * FARD) * m4::rotation_r_yxz(-camera_angles);
	auto camera_matrix = rotproj * m4::translation(-camera->position.local);

	frustum = create_frustum_planes_d3d(camera_matrix);

	struct BrushStroke {
		ChunkRelativePosition position;
		s32 radius;
		s32 strength;
	};
	List<BrushStroke> opponent_brush;
	opponent_brush.allocator = temporary_allocator;

	{
		while (auto maybe_message = with(net_queue_mutex, net_queue.pop())) {
			auto message = maybe_message.value_unchecked();
			switch (message.kind) {
				using enum NetMessageKind;
				case brush:
					print("brush: ");
					print("position: {} ", message.brush.position);
					print("radius: {} ", message.brush.radius);
					print("strength: {} ", message.brush.strength);
					print("\n");
					if (message.brush.radius > CHUNKW) {
						print("bad radius!\n");
						break;
					}

					opponent_brush.add({
						.position = message.brush.position,
						.radius   = message.brush.radius,
						.strength = message.brush.strength,
					});
					break;
				case set_position:
					opponent_position = message.set_position;
					break;
				//case set_chunk: {
				//	auto &chunk = saved_chunks.get_or_insert(message.set_chunk.position);
				//	memcpy(chunk.sdf, message.set_chunk.sdf, sizeof(chunk.sdf));
				//	print("received chunk {}\n", message.set_chunk.position);
				//	break;
				//}
				//case end_chunk_transfer:
				//	world_transfer_ended = true;
				//	print("received end_chunk_transfer\n");
				//	break;
				default:
					print("bad message! {}\n", (u8)message.kind);
					break;
			}
		}
	}
#if 0
	if (!is_server && !world_transfer_ended)
		return;
#endif

	send_message({
		.kind = NetMessageKind::set_position,
		.set_position = camera->position,
	});

	generate_chunks_around();

	auto camera_forward = camera_rotation * v3f{0,0,1};

	for (auto brush : opponent_brush) {
		add_sdf_sphere(brush.position, brush.radius, brush.strength);
	}
	if (auto hit = global_raycast(camera->position, camera_forward, CHUNKW*2)) {
		switch (belt_item_index) {
			case 1: {
				if (mouse_held(0) || mouse_held(1)) {
					auto position = hit.position + camera_forward * (brush_size - 1.5) * (mouse_held(0) ? 1 : -1);
					auto strength = brush_size * 32 * (mouse_held(0) ? 1 : -1);
					add_sdf_sphere_synchronized(position, brush_size, strength);
				}
				break;
			}
			case 2: {
				if (mouse_down(0) || mouse_down(1)) {
					auto crp = hit.position + hit.normal * (mouse_down(0) ? 1 : -1) * 0.001f;
					auto p = floor_to_int(crp.local);
					auto chunk = get_chunk(crp.chunk);

					if (!chunk->blocks)
						allocate_blocks(chunk);

					(*chunk->blocks)[p.x][p.y][p.z].solid = mouse_down(0);
					remesh_blocks(chunk);
					chunk->needs_saving = true;
				}
				break;
			}
		}
	}

	if (key_down('1')) { belt_item_index = 1; }
	if (key_down('2')) { belt_item_index = 2; }

	if (mouse_held(2)) {
		auto &grenade_entity = create_entity();
		grenade_entity.position = camera->position + camera_forward;

		auto &grenade = grenade_entity.add_component<GrenadeComponent>();
		grenade.timer = 3;

		auto &physics = grenade_entity.add_component<PhysicsComponent>();
		physics.set_velocity(camera_forward * 50);
		//physics.collision_box = {
		//	V3f(-.2),
		//	V3f(+.2),
		//};
	}

	if (key_held('C')) {
		auto chunk = get_chunk(camera->position.chunk);
		chunk_info_string = format(u8R"(chunk at {}
sdf_vb: {}
sdf_vb_vertex_count: {}
sdf_vb_vertex_offset: {}
grass_vb: {}
blocks_vb: {}
trees_instances_buffer: {}
sdf_vertex_positions.count: {}
sdf: {}
blocks: {}
lod_previous_frame: {}
frames_since_remesh: {}
average_sdf: {}
neighbor_mask: {}
sdf_generated: {}
sdf_mesh_generated: {}
block_mesh_generated: {}
has_surface: {}
needs_saving: {}
needs_filter_and_remesh: {}
sdf_at_interpolated: {}
)",
camera->position.chunk,
chunk->sdf_vb,
chunk->sdf_vb_vertex_count,
chunk->sdf_vb_vertex_offset,
chunk->grass_vb,
chunk->blocks_vb,
chunk->trees_instances_buffer,
chunk->sdf_vertex_positions.count,
chunk->sdf,
chunk->blocks,
chunk->lod_previous_frame,
chunk->frames_since_remesh,
chunk->average_sdf,
chunk->neighbor_mask,
chunk->sdf_generated,
chunk->sdf_mesh_generated,
chunk->block_mesh_generated,
chunk->has_surface,
chunk->needs_saving,
chunk->needs_filter_and_remesh,
sdf_at_interpolated(camera->position)
);
	}

	static bool update_allocations = false;
	if (key_down('L')) {
		update_allocations = !update_allocations;
	}

	if (update_allocations) {
		StringBuilder builder;
		defer { free(builder); };

		append(builder, "location: current | total\n");
		for_each(tracked_allocations, [&](auto location, auto info) {
			append_format(builder, "{}: {} | {}\n", location, format_bytes(info.current_size), format_bytes(info.total_size));
		});

		free(allocation_info_string);
		allocation_info_string = (List<utf8>)to_string(builder);
	}

	//if (key_down('G')) {
	//	for_each (chunks, [&](v3s chunk_position, Chunk chunk) {
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



	for (auto entity : entities_to_remove) {
		for (auto &component_index : entity->components) {
			switch (component_index.component_type) {
#define e(name) \
	case name##_index: \
		components<name>.at(component_index.component_index).on_destroy(); \
		components<name>.erase_at(component_index.component_index); \
		break;
ENUMERATE_COMPONENTS
#undef e
				default: invalid_code_path();
			}
		}
		entities.erase(entity);
	}
	entities_to_remove.clear();


	//
	// GRAPHICS
	//


	immediate_context->VSSetConstantBuffers(0, 1, &frame_cbuffer.cbuffer);
	immediate_context->PSSetConstantBuffers(0, 1, &frame_cbuffer.cbuffer);
	immediate_context->VSSetConstantBuffers(1, 1, &chunk_cbuffer.cbuffer);
	immediate_context->PSSetConstantBuffers(1, 1, &chunk_cbuffer.cbuffer);
	immediate_context->PSSetSamplers(DEFAULT_SAMPLER_SLOT, 1, &default_sampler_wrap);

	StaticList<Chunk *, pow3(DRAWD*2+1)> visible_chunks;

	u32 num_masks_skipped = 0;

	//
	// SHADOWS
	//
	{
		timed_block(profiler, profile_frame, "shadows");
		//frame_cbuffer.update({
		//	.mvp = camera_matrix,
		//	.rotproj = rotproj,
		//	.campos = camera->position.local,
		//	.ldir = normalize(v3f{1,3,2}),
		//});

		v3f light_angles = {pi/4, pi/6, 0};

		auto light_rotation = m4::rotation_r_zxy(light_angles);
		v3f light_dir = light_rotation * v3f{0,0,-1};

		f32 const shadow_pixels_in_meter = (f32)shadow_map_size / shadow_world_size / 2;

		auto lightr = m4::rotation_r_yxz(-light_angles);

		v3f lightpos = camera->position.to_v3f();
		lightpos = lightr * lightpos;
		lightpos *= shadow_pixels_in_meter;
		lightpos = round(lightpos);
		lightpos /= shadow_pixels_in_meter;
		lightpos = inverse(lightr) * lightpos;
		lightpos -= (v3f)(camera->position.chunk * CHUNKW);

		auto light_vp_matrix = m4::scale(1.f/v3f{shadow_world_size, shadow_world_size, CHUNKW*DRAWD}) * m4::rotation_r_yxz(-light_angles) * m4::translation(-lightpos);

		auto light_mvp = m4::translation(0,0,.5) * m4::scale(1,1,.5) * light_vp_matrix;

		frame_cbuffer.update({
			//.mvp = m4::scale(.1f * v3f{(f32)window_client_size.x / window_client_size.y, 1, 1}) * m4::rotation_r_yxz(-v3f{pi}),
			.mvp = light_mvp
		});
		auto light_frustum = create_frustum_planes_d3d(light_mvp);


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
		immediate_context->RSSetState(shadow_rasterizer);
		ID3D11ShaderResourceView *null = 0;
		immediate_context->PSSetShaderResources(SHADOW_TEXTURE_SLOT, 1, &null);
		immediate_context->OMSetRenderTargets(0, 0, shadow_dsv);
		immediate_context->OMSetBlendState(0, {}, -1);
		immediate_context->ClearDepthStencilView(shadow_dsv, D3D11_CLEAR_DEPTH, 1, 0);

		{
			timed_block(profiler, profile_frame, "sdf & blocks shadow");
#if 0
			for (
				auto pp_small = grid_map.begin();
				pp_small != grid_map.begin() + fully_meshed_chunk_index_end;
				++pp_small
			) {
				auto chunk_position = camera->position.chunk + (v3s)*pp_small;
				auto chunk = get_chunk(chunk_position);
#elif 0
			for (auto &chunk : flatten(chunks)) {
				auto chunk_position = get_chunk_position(&chunk);
#else
			for (u64 mask_index = 0; mask_index != count_of(nonempty_chunk_mask); ++mask_index) {
				auto mask = nonempty_chunk_mask[mask_index];
				if (mask == 0) {
					num_masks_skipped += 1;
					continue;
				}

				for (u64 bit_index = 0; bit_index != 64; ++bit_index) {
					if (mask & ((u64)1 << bit_index)) {
						auto chunk_position = get_chunk_position(mask_index * bits_in_chunk_mask + bit_index);
						auto chunk = get_chunk(chunk_position);
#endif
				auto relative_position = (v3f)((chunk_position-camera->position.chunk)*CHUNKW);
				if (contains_sphere(light_frustum, relative_position+V3f(CHUNKW/2), sqrt3*CHUNKW)) {
					visible_chunks.add(chunk);

					auto lod_index = debug_selected_lod == -1 ? get_chunk_lod_index(chunk_position) : debug_selected_lod;

					if (chunk->sdf_vb_vertex_count[lod_index]) {
						chunk_cbuffer.update({
							.relative_position = relative_position,
							.actual_position = (v3f)(chunk_position*CHUNKW),
							.vertex_offset = chunk->sdf_vb_vertex_offset[lod_index],
						});

						immediate_context->VSSetShaderResources(VERTEX_BUFFER_SLOT, 1, &chunk->sdf_vb);
#if USE_INDICES
						immediate_context->IASetIndexBuffer(chunk->index_buffer, DXGI_FORMAT_R32_UINT, 0);
						immediate_context->DrawIndexed(chunk->indices_count, 0, 0);
#else
						immediate_context->Draw(chunk->sdf_vb_vertex_count[lod_index], 0);
#endif

						if (chunk->blocks_vb.view) {
							immediate_context->VSSetShaderResources(VERTEX_BUFFER_SLOT, 1, &chunk->blocks_vb.view);
#if USE_INDICES
							immediate_context->IASetIndexBuffer(chunk->index_buffer, DXGI_FORMAT_R32_UINT, 0);
							immediate_context->DrawIndexed(chunk->indices_count, 0, 0);
#else
							immediate_context->Draw(chunk->blocks_vb.vertex_count, 0);
#endif
						}
					}
				}
					}}}
		}



		if (draw_trees) {
			timed_block(profiler, profile_frame, "tree shadow");
			immediate_context->RSSetState(no_cull_shadow_rasterizer);
			immediate_context->VSSetShader(tree_shadow_vs, 0, 0);
			immediate_context->PSSetShader(tree_shadow_ps, 0, 0);

			for (auto chunk : visible_chunks) {
				auto chunk_position = get_chunk_position(chunk);
				auto lod_index = get_chunk_lod_index(chunk_position);

				if (chunk->trees_instances_buffer.count) {
					auto relative_position = (v3f)((chunk_position-camera->position.chunk)*CHUNKW);
					chunk_cbuffer.update({
						.relative_position = relative_position,
						.actual_position = (v3f)(chunk_position*CHUNKW),
					});

					auto &model = tree_model.get_lod(lod_index);

					immediate_context->VSSetShaderResources(VERTEX_BUFFER_SLOT, 1, &model.vb);
					immediate_context->VSSetShaderResources(INSTANCE_BUFFER_SLOT, 1, &chunk->trees_instances_buffer.srv);
					immediate_context->PSSetShaderResources(ALBEDO_TEXTURE_SLOT, 1, &model.albedo);
					immediate_context->IASetIndexBuffer(model.ib, DXGI_FORMAT_R32_UINT, 0);
					immediate_context->RSSetState(model.no_cull ? no_cull_shadow_rasterizer : shadow_rasterizer);
					immediate_context->DrawIndexedInstanced(model.index_count, chunk->trees_instances_buffer.count, 0, 0, 0);
				}
			}
		}



		immediate_context->OMSetRenderTargets(0, 0, 0);
		immediate_context->PSSetShaderResources(SHADOW_TEXTURE_SLOT, 1, &shadow_srv);


		frame_cbuffer.update({
			.mvp = camera_matrix,
			.rotproj = rotproj,
			.light_vp_matrix = light_vp_matrix,
			.campos = camera->position.local,
			.time = (f32)get_time(the_timer),
			.ldir = light_dir,
			.frame = (f32)frame_number,
		});
	}

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
	{
		timed_block(profiler, profile_frame, "sky render");

		immediate_context->VSSetShader(sky_vs, 0, 0);
		immediate_context->PSSetShader(sky_ps, 0, 0);
		ID3D11ShaderResourceView *null = 0;
		immediate_context->PSSetShaderResources(SKY_TEXTURE_SLOT, 1, &null);
		immediate_context->RSSetState(0);
		immediate_context->OMSetRenderTargets(1, &sky_rt, 0);
		immediate_context->OMSetBlendState(0, {}, -1);
		immediate_context->Draw(36, 0);

		immediate_context->OMSetRenderTargets(1, &back_buffer, 0);
		immediate_context->ClearDepthStencilView(depth_stencil, D3D11_CLEAR_DEPTH, 1, 0);

		immediate_context->PSSetShaderResources(SKY_TEXTURE_SLOT, 1, &sky_srv);
	}

	//
	// SKY BLIT
	//
	{
		timed_block(profiler, profile_frame, "sky blit");
		immediate_context->VSSetShader(blit_vs, 0, 0);
		immediate_context->PSSetShader(blit_ps, 0, 0);
		immediate_context->RSSetState(0);
		immediate_context->OMSetRenderTargets(1, &back_buffer, 0);
		immediate_context->OMSetBlendState(0, {}, -1);
		immediate_context->Draw(6, 0);
	}

	//
	// CHUNKS SDF MESH
	//

	{
		timed_block(profiler, profile_frame, "draw far lands");
		immediate_context->OMSetRenderTargets(1, &back_buffer, depth_stencil);
		immediate_context->OMSetBlendState(0, {}, -1);
		immediate_context->VSSetShader(chunk_sdf_vs, 0, 0);
		immediate_context->PSSetShader(chunk_sdf_solid_ps, 0, 0);

		immediate_context->PSSetShaderResources(ALBEDO_TEXTURE_SLOT, 1, &ground_albedo);
		immediate_context->PSSetShaderResources(NORMAL_TEXTURE_SLOT, 1, &ground_normal);
		immediate_context->RSSetState(0);

		if (farlands_vb.vertex_count) {
			chunk_cbuffer.update({
				.relative_position = {},
				.vertex_offset = 0,
			});

			immediate_context->VSSetShaderResources(VERTEX_BUFFER_SLOT, 1, &farlands_vb.view);
			immediate_context->Draw(farlands_vb.vertex_count, 0);
		}
		immediate_context->ClearDepthStencilView(depth_stencil, D3D11_CLEAR_DEPTH, 1, 0);
	}

	visible_chunks.clear();
	{
		timed_block(profiler, profile_frame, "draw chunks sdf mesh");
		immediate_context->OMSetRenderTargets(1, &back_buffer, depth_stencil);
		immediate_context->OMSetBlendState(0, {}, -1);
		immediate_context->VSSetShader(chunk_sdf_vs, 0, 0);
		immediate_context->PSSetShader(chunk_sdf_solid_ps, 0, 0);

		immediate_context->PSSetShaderResources(ALBEDO_TEXTURE_SLOT, 1, &ground_albedo);
		immediate_context->PSSetShaderResources(NORMAL_TEXTURE_SLOT, 1, &ground_normal);
		immediate_context->RSSetState(0);

#if 0
		for (
			auto pp_small = grid_map.begin();
			pp_small != grid_map.begin() + fully_meshed_chunk_index_end;
			++pp_small
		) {
			auto chunk_position = camera->position.chunk + (v3s)*pp_small;
			auto chunk = get_chunk(chunk_position);
#elif 0
		for (auto &chunk : flatten(chunks)) {
			auto chunk_position = get_chunk_position(&chunk);
#else
		for (u64 mask_index = 0; mask_index != count_of(nonempty_chunk_mask); ++mask_index) {
			auto mask = nonempty_chunk_mask[mask_index];
			if (mask == 0)
				continue;

			for (u64 bit_index = 0; bit_index != 64; ++bit_index) {
				if (mask & ((u64)1 << bit_index)) {
					auto chunk_position = get_chunk_position(mask_index * bits_in_chunk_mask + bit_index);
					auto chunk = get_chunk(chunk_position);
#endif
			chunk->frames_since_remesh += 1;

			auto lod_index = debug_selected_lod == -1 ? get_chunk_lod_index(chunk_position) : debug_selected_lod;

			if (chunk->lod_previous_frame != lod_index) {
				chunk->previous_lod = chunk->lod_previous_frame;
				chunk->lod_previous_frame = lod_index;
				chunk->lod_t = 0;
			}

			// NOTE: If lod_t is zero, the shader doesn't not which way to blend.
			// So add frame time after setting it to 0.
			chunk->lod_t += frame_time;

			if (chunk_in_view(chunk_position)) {
				visible_chunks.add(chunk);

				if (chunk->sdf_vb_vertex_count[lod_index]) {
					auto relative_position = (v3f)((chunk_position-camera->position.chunk)*CHUNKW);
					chunk_cbuffer.update({
						.relative_position = relative_position,
						.was_remeshed = (f32)(chunk->frames_since_remesh < 5),
						.actual_position = (v3f)(chunk_position*CHUNKW),
						.vertex_offset = chunk->sdf_vb_vertex_offset[lod_index],
					});

					immediate_context->VSSetShaderResources(VERTEX_BUFFER_SLOT, 1, &chunk->sdf_vb);
#if USE_INDICES
					immediate_context->IASetIndexBuffer(chunk->index_buffer, DXGI_FORMAT_R32_UINT, 0);
					immediate_context->DrawIndexed(chunk->indices_count, 0, 0);
#else
					immediate_context->Draw(chunk->sdf_vb_vertex_count[lod_index], 0);
#endif
				}
			}
		}}}
	}

	//
	// CHUNKS BLOCKS MESH
	//
	{
		timed_block(profiler, profile_frame, "draw chunks blocks mesh");
		immediate_context->OMSetRenderTargets(1, &back_buffer, depth_stencil);
		immediate_context->OMSetBlendState(0, {}, -1);
		immediate_context->VSSetShader(chunk_block_vs, 0, 0);
		immediate_context->PSSetShader(chunk_block_ps, 0, 0);

		immediate_context->PSSetShaderResources(ALBEDO_TEXTURE_SLOT, 1, &planks_albedo);
		immediate_context->PSSetShaderResources(NORMAL_TEXTURE_SLOT, 1, &planks_normal);
		immediate_context->RSSetState(0);

		for (auto chunk : visible_chunks) {
			auto chunk_position = get_chunk_position(chunk);

			auto &vertex_buffer = chunk->blocks_vb;
			if (vertex_buffer.view) {
				auto relative_position = (v3f)((chunk_position-camera->position.chunk)*CHUNKW);
				chunk_cbuffer.update({
					.relative_position = relative_position,
					.actual_position = (v3f)(chunk_position*CHUNKW),
				});

				immediate_context->VSSetShaderResources(VERTEX_BUFFER_SLOT, 1, &vertex_buffer.view);
#if USE_INDICES
				immediate_context->IASetIndexBuffer(chunk->index_buffer, DXGI_FORMAT_R32_UINT, 0);
				immediate_context->DrawIndexed(chunk->indices_count, 0, 0);
#else
				immediate_context->Draw(vertex_buffer.vertex_count, 0);
#endif
			}
		}
	}

	//
	// CHUNKS GRASS
	//
	if (draw_grass) {
		timed_block(profiler, profile_frame, "draw chunks grass");
		immediate_context->RSSetState(no_cull_rasterizer);
		immediate_context->VSSetShader(grass_vs, 0, 0);
		immediate_context->PSSetShader(grass_ps, 0, 0);

		immediate_context->PSSetShaderResources(ALBEDO_TEXTURE_SLOT, 1, &grass_albedo);
		immediate_context->PSSetShaderResources(NORMAL_TEXTURE_SLOT, 1, &grass_normal);
		immediate_context->OMSetBlendState(0, {}, -1);

		for (auto &chunk : visible_chunks) {
			auto chunk_position = get_chunk_position(chunk);
			auto lod_index = get_chunk_lod_index(chunk_position);
			if (lod_index <= 1) {
				if (chunk->grass_vb.view) {
					auto relative_position = (v3f)((chunk_position-camera->position.chunk)*CHUNKW);
					chunk_cbuffer.update({
						.relative_position = relative_position,
						.actual_position = (v3f)(chunk_position*CHUNKW),
					});

					immediate_context->VSSetShaderResources(VERTEX_BUFFER_SLOT, 1, &chunk->grass_vb.view);
#if USE_INDICES
					immediate_context->IASetIndexBuffer(chunk->index_buffer, DXGI_FORMAT_R32_UINT, 0);
					immediate_context->DrawIndexed(chunk->indices_count, 0, 0);
#else
					immediate_context->Draw(chunk->grass_vb.vertex_count, 0);
#endif
				}
			}
		}
	}

	// TREE
	if (draw_trees) {
		timed_block(profiler, profile_frame, "trees surface");
		immediate_context->VSSetShader(tree_vs, 0, 0);
		immediate_context->PSSetShader(tree_ps, 0, 0);
		immediate_context->PSSetSamplers(DEFAULT_SAMPLER_SLOT, 1, &default_sampler_clamp);
		// immediate_context->OMSetBlendState(alpha_to_coverage_blend, {}, -1);

		for (auto &chunk : visible_chunks) {
			auto chunk_position = get_chunk_position(chunk);
			auto lod_index = get_chunk_lod_index(chunk_position);

			if (chunk->trees_instances_buffer.count) {
				auto relative_position = (v3f)((chunk_position-camera->position.chunk)*CHUNKW);
				chunk_cbuffer.update({
					.relative_position = relative_position,
					.actual_position = (v3f)(chunk_position*CHUNKW),
					.lod_t = chunk->lod_t,
				});

				auto &model = tree_model.get_lod(lod_index);
				immediate_context->VSSetShaderResources(VERTEX_BUFFER_SLOT, 1, &model.vb);
				immediate_context->VSSetShaderResources(INSTANCE_BUFFER_SLOT, 1, &chunk->trees_instances_buffer.srv);
				immediate_context->PSSetShaderResources(ALBEDO_TEXTURE_SLOT, 1, &model.albedo);
				immediate_context->PSSetShaderResources(NORMAL_TEXTURE_SLOT, 1, &model.normal);
				immediate_context->PSSetShaderResources(AO_TEXTURE_SLOT,     1, &model.ao);
				immediate_context->RSSetState(model.no_cull ? no_cull_rasterizer : 0);
				immediate_context->IASetIndexBuffer(model.ib, DXGI_FORMAT_R32_UINT, 0);
				immediate_context->DrawIndexedInstanced(model.index_count, chunk->trees_instances_buffer.count, 0, 0, 0);

				if (chunk->lod_t < 1) {
					chunk_cbuffer.update({
						.relative_position = relative_position,
						.actual_position = (v3f)(chunk_position*CHUNKW),
						.lod_t = -chunk->lod_t,
					});

					auto &model = tree_model.get_lod(chunk->previous_lod);
					immediate_context->VSSetShaderResources(VERTEX_BUFFER_SLOT, 1, &model.vb);
					immediate_context->VSSetShaderResources(INSTANCE_BUFFER_SLOT, 1, &chunk->trees_instances_buffer.srv);
					immediate_context->PSSetShaderResources(ALBEDO_TEXTURE_SLOT, 1, &model.albedo);
					immediate_context->PSSetShaderResources(NORMAL_TEXTURE_SLOT, 1, &model.normal);
					immediate_context->PSSetShaderResources(AO_TEXTURE_SLOT,     1, &model.ao);
					immediate_context->RSSetState(model.no_cull ? no_cull_rasterizer : 0);
					immediate_context->IASetIndexBuffer(model.ib, DXGI_FORMAT_R32_UINT, 0);
					immediate_context->DrawIndexedInstanced(model.index_count, chunk->trees_instances_buffer.count, 0, 0, 0);
				}
			}
		}
	}

	immediate_context->RSSetState(0);
	immediate_context->PSSetSamplers(DEFAULT_SAMPLER_SLOT, 1, &default_sampler_wrap);

	if (key_down('R'))
		wireframe_rasterizer_enabled = !wireframe_rasterizer_enabled;

	if (wireframe_rasterizer_enabled) {
		immediate_context->RSSetState(wireframe_rasterizer);
		immediate_context->VSSetShader(chunk_sdf_vs, 0, 0);
		immediate_context->PSSetShader(chunk_sdf_wire_ps, 0, 0);
		immediate_context->OMSetBlendState(alpha_blend, {}, -1);
		for (auto &chunk : visible_chunks) {
			auto chunk_position = get_chunk_position(chunk);
			auto lod_index = debug_selected_lod == -1 ? get_chunk_lod_index(chunk_position) : debug_selected_lod;
			if (chunk->sdf_vb_vertex_count[lod_index]) {
				auto relative_position = (v3f)((chunk_position-camera->position.chunk)*CHUNKW);
				chunk_cbuffer.update({
					.relative_position = relative_position,
					.actual_position = (v3f)(chunk_position*CHUNKW),
					.vertex_offset = chunk->sdf_vb_vertex_offset[lod_index],
				});

				immediate_context->VSSetShaderResources(VERTEX_BUFFER_SLOT, 1, &chunk->sdf_vb);
#if USE_INDICES
				immediate_context->IASetIndexBuffer(chunk->index_buffer, DXGI_FORMAT_R32_UINT, 0);
				immediate_context->DrawIndexed(chunk->indices_count, 0, 0);
#else
				immediate_context->Draw(chunk->sdf_vb_vertex_count[lod_index], 0);
#endif
			}
		}
	}

	immediate_context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
	immediate_context->VSSetShader(cursor_vs, 0, 0);
	immediate_context->PSSetShader(cursor_ps, 0, 0);

	auto draw_cursor = [&](ChunkRelativePosition position) {
		chunk_cbuffer.update({
			.relative_position = position.local + (v3f)((position.chunk-camera->position.chunk)*CHUNKW),
		});
		immediate_context->Draw(36, 0);
	};

	draw_cursor(opponent_position);

	for (auto &grenade : components<GrenadeComponent>) {
		draw_cursor(grenade.entity.position);
	}

	for (auto &ps : components<ParticleSystemComponent>) {
		for (auto &particle : ps.particles)
			draw_cursor({ps.particles_chunk_position, particle.position});
	}

	immediate_context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_LINELIST);
	immediate_context->RSSetState(0);
	immediate_context->OMSetRenderTargets(1, &back_buffer, 0);
	immediate_context->OMSetBlendState(0, {}, -1);
	immediate_context->VSSetShader(crosshair_vs, 0, 0);
	immediate_context->PSSetShader(crosshair_ps, 0, 0);
	immediate_context->Draw(4, 0);


	{
		timed_block(profiler, profile_frame, "ui");
		StringBuilder builder;
		builder.allocator = temporary_allocator;

		append_format(builder, u8R"(fps: {}
brush_size: {}
chunk_generation_amount_factor: {}
camera->position: {}
first_not_generated_sdf_chunk_index: {}
first_not_fully_meshed_chunk_index: {}
fully_meshed_chunk_index_end: {}
num_masks_skipped: {}
smoothed_average_generation_time: {} ms
generate_time: {} ms
sdfs_generated_per_frame: {}
n_sdfs_can_generate_this_frame: {}
remesh_count: {}
profile:
)"s,
smooth_fps,
brush_size,
chunk_generation_amount_factor,
camera->position,
first_not_generated_sdf_chunk_index,
first_not_fully_meshed_chunk_index,
fully_meshed_chunk_index_end,
num_masks_skipped,
smoothed_average_generation_time * 1000,
generate_time * 1000,
sdfs_generated_per_frame,
n_sdfs_can_generate_this_frame,
remesh_count
);
		HashMap<Span<utf8>, u64> time_spans;
		time_spans.allocator = temporary_allocator;

		for (auto &span : profiler.recorded_time_spans) {
			time_spans.get_or_insert(span.name) += span.end - span.begin;
		}
		for_each(time_spans, [&] (auto name, auto duration) {
			append_format(builder, "  {} {}us\n", name, duration * 1'000'000 / performance_frequency);
		});

		append_format(builder, "chunk info:\n{}", chunk_info_string);
		append_format(builder, "allocation info:\n{}", allocation_info_string);

		auto string = to_string(builder);
		defer { free(string); };

		draw_text((Span<utf8>)string, 16, {});
	}
#if 0
	for (auto &chunk : flatten(chunks)) {
		auto chunk_position = get_chunk_position(&chunk);
		auto &vertex_buffer = chunk.sdf_vertex_buffers[debug_selected_lod == -1 ? get_chunk_lod_index(chunk_position) : debug_selected_lod];
		if (vertex_buffer.view) {
			timed_block(profiler, profile_frame, "draw chunk");
			if (chunk_in_view(chunk_position)) {
				auto relative_position = (v3f)((chunk_position-camera->position.chunk)*CHUNKW);

				v4f p = camera_matrix * V4f(relative_position, 1);
				if (p.z >= 0) {
					p.xyz /= p.w;
					draw_text(to_string(
						FormatInt{.value=(chunk.neighbor_mask.x << 2) | (chunk.neighbor_mask.y << 1) | (chunk.neighbor_mask.z << 0), .radix=2,.leading_zero_count=3}
					), 16, p.xy, true);
				}
			}
		}
	}
#endif
}
void resize() {
	if (swap_chain) {
		if (back_buffer) {
			back_buffer->Release();
			depth_stencil->Release();
			sky_rt->Release();
			sky_srv->Release();
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
			};
			dhr(device->CreateDepthStencilView(tex, &desc, &depth_stencil));
		}

		{
			ID3D11Texture2D *tex;
			defer { tex->Release(); };
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
		case WM_KILLFOCUS: {
			unlock_cursor();
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
	key_state[key] |= KeyState_up;
};
auto on_mouse_down = [](u8 button){
	lock_cursor();
	key_state[256 + button] = KeyState_down | KeyState_held;
};
auto on_mouse_up = [](u8 button){
	key_state[256 + button] |= KeyState_up;
};

template <class Write>
void encode_int32_multibyte(u32 x, Write &&write) requires requires { write((u8)0); } {
	while (x >= 0x80) {
		write((u8)((x & 0x7F) | 0x80));
		x >>= 7;
	}
	write((u8)x);
}

template <class Read>
u32 decode_int32_multibyte(Read &&read) requires requires { (u8)read(); } {
	u32 v = 0;
	u32 i = 0;
	u8 p;
	do {
		p = read();
		v |= (p & 0x7f) << i;
		i += 7;
	} while (p & 0x80);
	return v;
}

void receive_loop() {
	another_thread_is_connected = true;
	defer { another_thread_is_connected = false; };
	defer { print("exited receive loop\n"); };

	StaticRingQueue<u8, 1*MiB> queue;
	u8 buf[queue.capacity];

	try {
		while (1) {
			auto read = [&]<class T>(T &v) {
				u32 iters = 0;
				while (queue.count < sizeof(T)) {
					auto received_bytes = net::receive(opponent_socket, buf, queue.capacity - queue.count);
					if (received_bytes == -1) {
						print("client disconnected badly\n");
						print_wsa_error();
						throw 0;
					}
					if (received_bytes == 0) {
						if (iters == 256) {
							print("network error: not enough data\n");
							throw 1;
						}
						iters++;
						Sleep(1);
					} else {
						queue.push(Span(buf, received_bytes));
					}
				}

				auto p = (u8 *)&v;
				for (u32 i = 0; i < sizeof(T); ++i) {
					p[i] = queue.pop().value_unchecked();
				}
			};

			NetMessage message;

			read(message.kind);

			switch (message.kind) {
				using enum NetMessageKind;
#define C(x) case x: read(message.x); break;

				C(brush)
				C(set_position)
				//C(set_chunk)
				//C(end_chunk_transfer)

#undef C
				default:
					print("bad message: {}\n", (u8)message.kind);
					break;

			}

			scoped_lock(net_queue_mutex);
			net_queue.push(message);
		}
	} catch (int) {
	}
}


int main(int argc, char **argv) {
	// {
	// 	m4 a = *(m4 *)argv[0];
	// 	m4 b = *(m4 *)argv[1];
	// 	m4 c = a * b;
	// 	*(m4 *)argv[1] = c;
	// }

	HINSTANCE hInstance = GetModuleHandleW(0);
	init_allocator();

	init_tracking_allocator();
	current_allocator = tracking_allocator;
	tracking_allocator_fallback = os_allocator;

	void *test = current_allocator.allocate(16);
	void *test2 = current_allocator.reallocate(test, 16, 32);
	current_allocator.free(test2);

	executable_path = get_executable_path();
	executable_directory = parse_path(executable_path).directory;

	_chunks = (decltype(_chunks))VirtualAlloc(0, sizeof(chunks[0][0][0]) * pow3(DRAWD*2+1), MEM_COMMIT|MEM_RESERVE, PAGE_READWRITE);
	for (auto &chunk : flatten(chunks)) {
		construct(chunk.sdf_vertex_positions);
		chunk.lod_t = 1;
#if USE_INDICES
		construct(chunk.indices);
#endif
	}

	construct(profiler);
	construct(thread_pool);
	construct(entities);
	construct(saved_chunks);
	construct(net_queue);
	construct(entities_to_remove);
	construct(entities);
	construct(tree_model);
#define e(name) construct(components<name>);
ENUMERATE_COMPONENTS
#undef e

	thread_count = get_cpu_info().logical_processor_count;
	//thread_count = 1;
	init_thread_pool(thread_pool, thread_count-1); // main thread is also a worker

	init_printer();

	WSADATA wsaData;
	if (auto err = WSAStartup(MAKEWORD(2, 2), &wsaData); err != 0) {
		printf("WSAStartup failed: %d\n", err);
		exit(1);
	}
	defer { WSACleanup(); };

	// TODO: WTF?
	// msvc optimized this to just int3
	// this was good before and wasn't changed at all.
#if 0
	StaticRingQueue<u8, 4> test;
	test.push(0);
	test.push(1);
	assert(test.pop().value() == 0);
	assert(test.pop().value() == 1);
	test.push(2);
	test.push(3);
	test.push(4);
	test.push(5);
	assert(test.pop().value() == 2);
	assert(test.pop().value() == 3);
	assert(test.pop().value() == 4);
	assert(test.pop().value() == 5);
	test.push(as_span({(u8)6,(u8)7,(u8)8,(u8)9}));
	assert(test.pop().value() == 6);
	assert(test.pop().value() == 7);
	assert(test.pop().value() == 8);
	assert(test.pop().value() == 9);
	test.push(0);
	assert(test.pop().value() == 0);
	test.push(as_span({(u8)6,(u8)7,(u8)8,(u8)9}));
	assert(test.pop().value() == 6);
	assert(test.pop().value() == 7);
	assert(test.pop().value() == 8);
	assert(test.pop().value() == 9);
#endif
	if (argc != 1) {
		opponent_socket = net::create_socket(net::Connection_tcp);

		if (!net::connect(opponent_socket, inet_addr(argv[1]), 27015) != 0) {
			print("connection failed");
			print_wsa_error();
			exit(1);
		}

		create_thread([] (Thread *) {
			receive_loop();
		}, 0);
	} else {
		is_server = true;
		create_thread([] (Thread *) {

			auto listener = net::create_socket(net::Connection_tcp);
			if (!listener) {
				printf("Error at socket(): %ld\n", WSAGetLastError());
				exit(1);
			}
			defer { net::close(listener); };

			net::bind(listener, 27015);

			net::listen(listener, 1);

			SOCKADDR_IN client;
			char host[NI_MAXHOST]{};
			char service[NI_MAXSERV]{};
			int clientSize = sizeof(client);
			print("accepting...\n");
			opponent_socket = (net::Socket)accept((SOCKET)listener, (SOCKADDR*)&client, &clientSize);
			if ((SOCKET)opponent_socket == INVALID_SOCKET) {
				print("accept failed\n");
				print_wsa_error();
				exit(1);
			}
			if (getnameinfo((SOCKADDR*)&client, sizeof(client), host, NI_MAXHOST, service, NI_MAXSERV, 0) == 0) {
				print("{} connected on port {}\n", host, service);
			} else {
				inet_ntop(AF_INET, &client.sin_addr, host, NI_MAXHOST);
				print("{} connected on port {}\n", host, ntohs(client.sin_port));
			}


			receive_loop();
		}, (void *)0);
	}

	auto save_path = format(u8"{}/world.save", executable_directory);

	if (is_server) {
		auto file = open_file(save_path, {.read=true});
		if (is_valid(file)) {
			defer { close(file); };

			auto map = map_file(file);
			defer { unmap_file(map); };

			auto chunk_count = *(u32 *)map.data.data;

			auto positions = (v3s *)(map.data.data + sizeof(u32));

			auto cursor    = (u8 *)(positions + chunk_count);

#define read(name, type) \
	if (cursor + sizeof(type) > map.data.end()) { \
		with(ConsoleColor::red, print("Failed to load save file: too little data\n")); \
		saved_chunks.clear(); \
		goto skip_load; \
	} \
	auto name = *(type *)cursor; \
	cursor += sizeof(type);

#define reade(name) \
	if (cursor + sizeof(name) > map.data.end()) { \
		with(ConsoleColor::red, print("Failed to load save file: too little data\n")); \
		saved_chunks.clear(); \
		goto skip_load; \
	} \
	memcpy(&name, cursor, sizeof(name)); \
	cursor += sizeof(name);

			for (u32 chunk_index = 0; chunk_index < chunk_count; ++chunk_index) {
				auto &saved = saved_chunks.get_or_insert(positions[chunk_index]);
				u32 offset = 0;
				while (offset < CHUNKW*CHUNKW*CHUNKW) {
					u32 count = 0;

					u32 i = 0;
					u8 count_part;
					do {
						reade(count_part);
						count |= (u32)(count_part & 0x7F) << i;
						i += 7;
					} while (count_part & 0x80);

					if (offset+count > CHUNKW*CHUNKW*CHUNKW) {
						with(ConsoleColor::red, print("Failed to load save file: chunk size corrupt\n"));
						saved_chunks.clear();
						goto skip_load;
					}

					read(sdf, u8);

					memset((s8 *)saved.sdf + offset, sdf, count);
					offset += count;
				}
				if (offset != CHUNKW*CHUNKW*CHUNKW) {
					with(ConsoleColor::red, print("Failed to load save file: chunk size in file is wrong\n"));
					saved_chunks.clear();
					goto skip_load;
				}

				reade(saved.blocks);
			}
		}
	}
skip_load:

	defer {
		if (!is_server)
			return;

		profiler.reset();
		defer { write_entire_file("save.tmd"s, as_bytes(profiler.output_for_timed())); };

		timed_block(profiler, true, "save");
		{
			timed_block(profiler, true, "write modified chunks");
			for (auto &chunk : flatten(chunks)) {
				if (chunk.needs_saving) {
					save_chunk(&chunk, get_chunk_position(&chunk));
				}
			}
		}

		u32 fail_count = 0;
		u32 const max_fail_count = 4;

		auto tmp_path = format(u8"{}.tmp", save_path);

		StringBuilder builder;
		auto write = [&](auto const &value) {
			append(builder, value_as_bytes(value));
		};

		write((u32)saved_chunks.total_value_count);

		for (auto &[position, saved] : saved_chunks) {
			write((v3s)position);
		}
		for (auto &[position, saved] : saved_chunks) {
			timed_block(profiler, true, "write chunk");

			auto sdf = flatten(saved.sdf);
			u32 first_index = 0;
			while (first_index < CHUNKW*CHUNKW*CHUNKW) {
				auto first = sdf[first_index];
				u32 end_index = first_index + 1;
				for (; end_index < sdf.count; ++end_index) {
					if (sdf[end_index] == first) {
						continue;
					} else {
						break;
					}
				}

				auto count = end_index - first_index;

				encode_int32_multibyte(count, [&](u8 b){write(b);});

				write((s8)first);

				first_index += count;
			}
			assert(first_index == CHUNKW*CHUNKW*CHUNKW);

			append(builder, value_as_bytes(saved.blocks));
		}

		timed_block(profiler, true, "write file");
		write_entire_file(save_path, as_bytes(to_string(builder)));
	};

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

	resize();

	chunk_sdf_vs = create_vs(HLSL_CBUFFER HLSL_COMMON R"(
struct Vertex {
	float3 position;
	uint normal;
};

StructuredBuffer<Vertex> s_vertex_buffer : VERTEX_BUFFER_SLOT;

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
	float3 pos = s_vertex_buffer[vertex_id+c_vertex_offset].position + c_relative_position;
	wpos = pos;
	normal = decode_normal(s_vertex_buffer[vertex_id+c_vertex_offset].normal);
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
		color = float3(1, 0, 0);
		//color = float3(0,0,1);
	}

	position = mul(c_mvp, float4(pos, 1.0f));
	screen_uv = position * float4(1, -1, 1, 1);

	shadow_map_uv = mul(light_vp_matrix, float4(pos, 1));
}
)"s);

	chunk_sdf_solid_ps = create_ps(HLSL_CBUFFER HLSL_COMMON HLSL_LIGHTING HLSL_TRIPLANAR R"(
void main(
	in float3 normal : NORMAL,
	in float3 color_ : COLOR,
	in float3 wpos : WPOS,
	in float3 view : VIEW,
	in float4 screen_uv : SCREEN_UV,
	in float4 shadow_map_uv : SHADOW_MAP_UV,

	out float4 pixel_color : SV_Target
) {
	normal = normalize(normal);

	float3 L = c_ldir;
	float3 N = float4(triplanar_normal(normal_texture, default_sampler, wpos/32, normal), 1);

	float3 V = -normalize(view);
	float3 H = normalize(V + L);

	float NV = max(dot(N, V), 1e-3f);
	float NL = max(dot(N, L), 1e-3f);
	float NH = max(dot(N, H), 1e-3f);
	float VH = max(dot(V, H), 1e-3f);

	float4 trip = triplanar(albedo_texture, default_sampler, wpos/32, normal);
	//pixel_color = N.xyzz;
	//return;

	float3 grass = float3(.6,.9,.2)*.8;
	float3 rock  = float3(.2,.2,.1);

	float gr = smoothstep(0.3, 0.7, normal.y);
	gr = lerp(gr, trip, (0.5f - abs(0.5f - gr)) * 2);

	float3 albedo = lerp(rock, grass, gr) * trip;// * color_;

	float metalness = 0;
	float roughness = 1;

	screen_uv = (screen_uv / screen_uv.w) * 0.5 + 0.5;
	float3 ambient_color = sky_texture.Sample(default_sampler, screen_uv.xy);

	float3 F0 = 0.04;
	F0 = lerp(F0, albedo, metalness);

	float D = trowbridge_reitz_ggx(NH, roughness);
	float G = smith_schlick(NV, NL, k_direct(roughness));
	float3 F = fresnel_schlick(NV, F0);
	float3 specular = cook_torrance(D, F, G, NV, NL);
	float3 diffuse = albedo * NL * (1 - metalness) / pi * (1 - specular);

	float3 ambient_specular = ambient_color * F * smith_schlick(NV, 1, k_ibl(roughness));
	float3 ambient_diffuse = albedo * ambient_color * (1-ambient_specular);
	float3 ambient = (ambient_diffuse + ambient_specular) / pi;

	shadow_map_uv /= shadow_map_uv.w;
	shadow_map_uv.y *= -1;
	float shadow_mask = saturate(map(length(shadow_map_uv.xyz), 0.9, 1, 0, 1));
	shadow_map_uv = shadow_map_uv * 0.5 + 0.5;

	float lightness = lerp(shadow_texture.SampleCmpLevelZero(shadow_sampler, shadow_map_uv.xy, shadow_map_uv.z).x, 1, shadow_mask);
	pixel_color.rgb = ambient + (diffuse + specular) * lightness;
	pixel_color.a = 1;

	float fog = min(1, length(view) / (CHUNKW*FARD));
	fog *= fog;
	fog *= fog;
	pixel_color.rgb = lerp(pixel_color.rgb, ambient_color, fog);

	//if (frac(wpos/32).x*32 < 1 ||
	//	frac(wpos/32).y*32 < 1 ||
	//	frac(wpos/32).z*32 < 1
	//)
	//	pixel_color = float4(1,0,0,1);

	//pixel_color = shadowtex.SampleCmpLevelZero(dsam, shadow_map_uv.xy, shadow_map_uv.z-0.01);
	//pixel_color = shadowtex.Sample(sam, shadow_map_uv.xy);

}
)"s);


	chunk_sdf_wire_ps = create_ps(R"(
void main(in float3 normal : NORMAL, in float3 color : COLOR, out float4 pixel_color : SV_Target) {
	pixel_color.rgb = 0;
	pixel_color.a = 1;
}
)"s);

	cursor_vs = create_vs(HLSL_CBUFFER R"(
void main(in uint vertex_id : SV_VertexID, out float3 color : COLOR, out float4 position : SV_Position) {
	float3 positions[] = {
		// front
		{-1, 1, 1},
		{-1,-1, 1},
		{ 1,-1, 1},
		{ 1, 1, 1},
		{-1, 1, 1},
		{ 1,-1, 1},

		// back
		{-1,-1,-1},
		{-1, 1,-1},
		{ 1,-1,-1},
		{-1, 1,-1},
		{ 1, 1,-1},
		{ 1,-1,-1},

		// right
		{ 1,-1, 1},
		{ 1,-1,-1},
		{ 1, 1,-1},
		{ 1, 1, 1},
		{ 1,-1, 1},
		{ 1, 1,-1},

		// left
		{-1,-1,-1},
		{-1,-1, 1},
		{-1, 1,-1},
		{-1,-1, 1},
		{-1, 1, 1},
		{-1, 1,-1},

		// top
		{-1, 1,-1},
		{-1, 1, 1},
		{ 1, 1,-1},
		{-1, 1, 1},
		{ 1, 1, 1},
		{ 1, 1,-1},

		// bottom
		{-1,-1, 1},
		{-1,-1,-1},
		{ 1,-1,-1},
		{ 1,-1, 1},
		{-1,-1, 1},
		{ 1,-1,-1},
	};

	position = mul(c_mvp, float4(positions[vertex_id]+c_relative_position, 1.0f));
	color = positions[vertex_id];
}
)"s);
	cursor_ps = create_ps(R"(
void main(in float3 color: COLOR, out float4 pixel_color : SV_Target) {
	pixel_color = float4(color, 1);
}
)"s);

	sky_vs = create_vs(HLSL_CBUFFER R"(
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
	sky_ps = create_ps(HLSL_CBUFFER HLSL_COMMON R"(
void main(in float3 view : VIEW, out float4 pixel_color : SV_Target) {
	float3 L = c_ldir;
	float3 V = normalize(view);
	float3 color = 0;
	color = lerp(float3(.27,.34,.37), float3(.01,.10,.8), smoothstep(-1, 1, dot(V, float3(0,1,0))));
	//color = lerp(0, color, smoothstep(-0.5, 0, dot(V, float3(0,1,0))));
	color += pow(map_clamped(dot(V, L), .5, .99, 0, 1), 100);
	pixel_color = float4(rgb_to_srgb(color), 1);
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
	blit_ps = create_ps(HLSL_CBUFFER R"(
void main(in float2 uv : UV, out float4 pixel_color : SV_Target) {
	pixel_color = sky_texture.Sample(default_sampler, uv);
}
)"s);

	shadow_vs = create_vs(HLSL_CBUFFER R"(
struct Vertex {
	float3 position;
	uint normal;
};

StructuredBuffer<Vertex> s_vertex_buffer : VERTEX_BUFFER_SLOT;

void main(
	in uint vertex_id : SV_VertexID,
	out float4 screen_uv : SCREEN_UV,
	out float4 position : SV_Position
) {
	float3 pos = s_vertex_buffer[vertex_id+c_vertex_offset].position + c_relative_position;
	position = mul(c_mvp, float4(pos, 1.0f));
	screen_uv = position * float4(1, -1, 1, 1);
}
)"s);

	shadow_ps = create_ps(HLSL_CBUFFER HLSL_COMMON R"(
void main(
	in float4 screen_uv : SCREEN_UV
) {
	//float dx = ddx(screen_uv.z);
	//float dy = ddy(screen_uv.z);
	//pixel_color = float2(screen_uv.z, dx*dx + dy*dy);//map(screen_uv.z, -1, 1, 0, 1);
}
)"s);

	font_vs = create_vs(R"(
struct Vertex {
	float2 position;
	float2 uv;
};

StructuredBuffer<Vertex> vertices : VERTEX_BUFFER_SLOT;

void main(in uint vertex_id : SV_VertexID, out float2 uv : UV, out float4 position : SV_Position) {
	Vertex v = vertices[vertex_id];
	position = float4(v.position, 0, 1);
	uv = v.uv;
}
)"s);
	font_ps = create_ps(HLSL_CBUFFER R"(
Texture2D tex : register(t0);
void main(in float2 uv : UV, out float4 pixel_color0 : SV_Target, out float4 pixel_color1 : SV_Target1) {
	pixel_color0 = 1;
	pixel_color1 = float4(tex.Sample(default_sampler, uv).rgb, 1);
}
)"s);

	crosshair_vs = create_vs(R"(
void main(in uint vertex_id : SV_VertexID, out float4 position : SV_Position) {
	float size = .01;
	float2 positions[] = {
		{-size, 0},
		{ size, 0},
		{ 0,-size},
		{ 0, size},
	};

	position = float4(positions[vertex_id], 0, 1);
}
)"s);
	crosshair_ps = create_ps(R"(
void main(out float4 pixel_color : SV_Target) {
	pixel_color = 1;
}
)"s);

	grass_vs = create_vs(HLSL_CBUFFER HLSL_COMMON R"(
struct Vertex {
	float3 origin;
	float3 position;
	float3 normal;
	float2 uv;
};

StructuredBuffer<Vertex> s_vertex_buffer : VERTEX_BUFFER_SLOT;

void main(
	in uint vertex_id : SV_VertexID,

	out float2 uv : UV,
	out float3 normal : NORMAL,
	out float3 wpos : WPOS,
	out float3 view : VIEW,
	out float4 screen_uv : SCREEN_UV,
	out float4 shadow_map_uv : SHADOW_MAP_UV,
	out float4 position : SV_Position
) {
	Vertex v = s_vertex_buffer[vertex_id];
	float3 pos = lerp(v.position, v.origin, min(1,distance(c_campos, v.origin+c_relative_position)/(CHUNKW*3))) + c_relative_position;


	wpos = pos/8;
	uv = v.uv;
	normal = v.normal;
	view = pos - c_campos;

	position = mul(c_mvp, float4(pos, 1.0f));
	screen_uv = position * float4(1, -1, 1, 1);

	shadow_map_uv = mul(light_vp_matrix, float4(pos, 1));
}
)"s);

	grass_ps = create_ps(HLSL_CBUFFER HLSL_COMMON HLSL_LIGHTING HLSL_TRIPLANAR R"(
void main(
	in float2 uv : UV,
	in float3 normal : NORMAL,
	in float3 wpos : WPOS,
	in float3 view : VIEW,
	in float4 screen_uv : SCREEN_UV,
	in float4 shadow_map_uv : SHADOW_MAP_UV,

	out float4 pixel_color : SV_Target
) {
	float4 colortex = albedo_texture.Sample(default_sampler, uv);

	clip(colortex.a-0.5f);

	float3 albedo = colortex.rgb * float3(.6,.9,.2)*.8;


	float3 L = c_ldir;
	float3 N = normal;

	float3 V = -normalize(view);
	float3 H = normalize(V + L);

	float NV = max(dot(N, V), 0.25f);
	float NL = max(dot(N, L), 1e-3f);
	float NH = max(dot(N, H), 1e-3f);
	float VH = max(dot(V, H), 1e-3f);

	float metalness = 0;
	float roughness = 1;

	screen_uv = (screen_uv / screen_uv.w) * 0.5 + 0.5;
	float3 ambient_color = sky_texture.Sample(default_sampler, screen_uv.xy);

	float3 F0 = 0.04;
	F0 = lerp(F0, albedo, metalness);

	float D = trowbridge_reitz_ggx(NH, roughness);
	float G = smith_schlick(NV, NL, k_direct(roughness));
	float3 F = fresnel_schlick(NV, F0);
	float3 specular = cook_torrance(D, F, G, NV, NL);
	float3 diffuse = albedo * NL * (1 - metalness) / pi * (1 - specular);

	float3 ambient_specular = ambient_color * F * smith_schlick(NV, 1, k_ibl(roughness));
	float3 ambient_diffuse = albedo * ambient_color * (1-ambient_specular);
	float3 ambient = (ambient_diffuse + ambient_specular) / pi;


	shadow_map_uv /= shadow_map_uv.w;
	shadow_map_uv.y *= -1;
	float shadow_mask = saturate(map(length(shadow_map_uv.xyz), 0.9, 1, 0, 1));
	shadow_map_uv = shadow_map_uv * 0.5 + 0.5;

	float lightness = lerp(shadow_texture.SampleCmpLevelZero(shadow_sampler, shadow_map_uv.xy, shadow_map_uv.z).x, 1, shadow_mask);
	pixel_color.rgb = ambient + (diffuse + specular) * lightness;
	pixel_color.a = 1;

	float fog = min(1, length(view) / (CHUNKW*FARD));
	fog *= fog;
	fog *= fog;
	pixel_color.rgb = lerp(pixel_color.rgb, ambient_color, fog);

	//pixel_color = shadowtex.SampleCmpLevelZero(dsam, shadow_map_uv.xy, shadow_map_uv.z-0.01);
	//pixel_color = shadowtex.Sample(sam, shadow_map_uv.xy);

}
)"s);

	chunk_block_vs = create_vs(HLSL_CBUFFER HLSL_COMMON R"(
struct Vertex {
	float3 position;
	uint normal;
};

StructuredBuffer<Vertex> s_vertex_buffer : VERTEX_BUFFER_SLOT;

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
	wpos = pos;
	normal = decode_normal(s_vertex_buffer[vertex_id].normal);
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
		color = 1;
		//color = float3(0,0,1);
	}

	position = mul(c_mvp, float4(pos, 1.0f));
	screen_uv = position * float4(1, -1, 1, 1);

	shadow_map_uv = mul(light_vp_matrix, float4(pos, 1));
}
)"s);

	chunk_block_ps = create_ps(HLSL_CBUFFER HLSL_COMMON HLSL_LIGHTING HLSL_TRIPLANAR R"(
void main(
	in float3 normal : NORMAL,
	in float3 color_ : COLOR,
	in float3 wpos : WPOS,
	in float3 view : VIEW,
	in float4 screen_uv : SCREEN_UV,
	in float4 shadow_map_uv : SHADOW_MAP_UV,

	out float4 pixel_color : SV_Target
) {
	normal = normalize(normal);

	float3 L = c_ldir;
	float3 N = float4(triplanar_normal(normal_texture, default_sampler, wpos, normal), 1);

	float3 V = -normalize(view);
	float3 H = normalize(V + L);

	float NV = max(dot(N, V), 1e-3f);
	float NL = max(dot(N, L), 1e-3f);
	float NH = max(dot(N, H), 1e-3f);
	float VH = max(dot(V, H), 1e-3f);

	float3 albedo = triplanar(albedo_texture, default_sampler, wpos, normal).xyz;

	float metalness = 0;
	float roughness = 1;

	screen_uv = (screen_uv / screen_uv.w) * 0.5 + 0.5;
	float3 ambient_color = sky_texture.Sample(default_sampler, screen_uv.xy);

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

	float lightness = lerp(shadow_texture.SampleCmpLevelZero(shadow_sampler, shadow_map_uv.xy, shadow_map_uv.z).x, 1, shadow_mask);
	pixel_color.rgb = ambient + (diffuse + specular) * lightness;
	pixel_color.a = 1;

	float fog = min(1, length(view) / (CHUNKW*FARD));
	fog *= fog;
	fog *= fog;
	pixel_color.rgb = lerp(pixel_color.rgb, ambient_color, fog);

	//pixel_color = shadowtex.SampleCmpLevelZero(dsam, shadow_map_uv.xy, shadow_map_uv.z-0.01);
	//pixel_color = shadowtex.Sample(sam, shadow_map_uv.xy);

}
)"s);

	tree_vs = create_vs(HLSL_CBUFFER R"(
struct Vertex {
	float3 position;
	float3 normal;
	float4 tangent;
	float4 color;
	float2 uv;
};

StructuredBuffer<Vertex> s_vertex_buffer : VERTEX_BUFFER_SLOT;

struct Instance {
	float3x3 mat;
	float3 position;
};

StructuredBuffer<Instance> s_instance_buffer : INSTANCE_BUFFER_SLOT;

void main(
	in uint vertex_id : SV_VertexID,
	in uint instance_id : SV_InstanceID,

	out float3 normal : NORMAL,
	out float4 tangent : TANGENT,
	out float3 color : COLOR,
	out float2 uv : UV,
	out float3 wpos : WPOS,
	out float3 view : VIEW,
	out float4 screen_uv : SCREEN_UV,
	out float4 shadow_map_uv : SHADOW_MAP_UV,
	out float4 position : SV_Position
) {
	Instance instance = s_instance_buffer[instance_id];
	float3 pos = mul(instance.mat, s_vertex_buffer[vertex_id].position) + instance.position + c_relative_position;

	wpos = pos;
	normal = mul(instance.mat, float4(s_vertex_buffer[vertex_id].normal, 0)).xyz;
	tangent = s_vertex_buffer[vertex_id].tangent;
	view = pos - c_campos;

	uv = s_vertex_buffer[vertex_id].uv;

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
		color = 1;
		//color = float3(0,0,1);
	}

	position = mul(c_mvp, float4(pos, 1.0f));
	screen_uv = position * float4(1, -1, 1, 1);

	shadow_map_uv = mul(light_vp_matrix, float4(pos, 1));
}
)"s);

	tree_ps = create_ps(HLSL_CBUFFER HLSL_COMMON HLSL_LIGHTING HLSL_TRIPLANAR R"(
void main(
	in float3 normal : NORMAL,
	in float4 tangent : TANGENT,
	in float3 color_ : COLOR,
	in float2 uv : UV,
	in float3 wpos : WPOS,
	in float3 view : VIEW,
	in float4 screen_uv : SCREEN_UV,
	in float4 shadow_map_uv : SHADOW_MAP_UV,
	in float4 pixel_position : SV_Position,

	in bool vface : SV_IsFrontFace,

	out float4 pixel_color : SV_Target
) {
	//clip(sign(c_lod_t) * (abs(c_lod_t) - lod_mask_texture.Sample(nearest_sampler, float3(pixel_position.xy,c_frame)/16).x));
	clip(sign(c_lod_t) * (abs(c_lod_t) - lod_mask_texture.Sample(nearest_sampler, wpos).x));
	//clip(sign(c_lod_t) * (abs(c_lod_t) - lod_mask_texture.Sample(nearest_sampler, float3(pixel_position.xy/16, 0).zxy).x));

	normal = normalize(normal);

	// FIXME: tangent is broken?

	// // derivations of the fragment position
	// float3 pos_dx = ddx( pixel_position );
	// float3 pos_dy = ddy( pixel_position );
	// // derivations of the texture coordinate
	// float2 texC_dx = ddx( uv );
	// float2 texC_dy = ddy( uv );
	// // tangent vector and binormal vector
	// float3 t = normalize(texC_dy.y * pos_dx - texC_dx.y * pos_dy);
	// float3 b = normalize(texC_dx.x * pos_dy - texC_dy.x * pos_dx);

	float3 L = c_ldir;
	float3 N = world_normal(normalize(normal), tangent, normal_texture.Sample(default_sampler, uv), 1);
	//N = lerp(N, L, 0.5);

	float3 V = -normalize(view);
	float3 H = normalize(V + L);

	float NV = max(dot(N, V), 1e-3f);
	float NL = max(dot(N, L), 1e-3f);
	float NH = max(dot(N, H), 1e-3f);
	float VH = max(dot(V, H), 1e-3f);

#if 0
	float3 data = albedo_texture.Sample(default_sampler, uv);
	pixel_color.a = map_clamped(data.x, .7, .6, 0, 1) + data.z;

	float3 albedo = lerp(
		lerp(float3(.03,.12,.01), float3(.37,.80,.19),data.y),
		lerp(float3(.12,.04,.01), float3(.80,.42,.19),data.y),
		data.z
	);
#else
	float4 data = albedo_texture.Sample(default_sampler, uv);
	pixel_color.a = data.a * 2;
	clip(pixel_color.a - 0.5);
	float3 albedo = data.rgb;
#endif

	//pixel_color = float4(tangent.xyz * float3(-1,1,-1), 1);
	//pixel_color = float4(N, 1);
	//return;


	float metalness = 0;
	float roughness = 1;

	screen_uv = (screen_uv / screen_uv.w) * 0.5 + 0.5;
	float3 ambient_color = sky_texture.Sample(default_sampler, screen_uv.xy);

	float3 F0 = 0.04;
	F0 = lerp(F0, albedo, metalness);

	float D = trowbridge_reitz_ggx(NH, roughness);
	float G = smith_schlick(NV, NL, k_direct(roughness));
	float3 F = fresnel_schlick(NV, F0);
	float3 specular = 0;//cook_torrance(D, F, G, NV, NL);
	float3 diffuse = albedo * NL * (1 - metalness) / pi;// * (1 - specular);

	float3 ambient_specular = 0;//ambient_color * F * smith_schlick(NV, 1, k_ibl(roughness));
	float3 ambient_diffuse = albedo * ambient_color * (1-ambient_specular);
	float3 ambient = (ambient_diffuse / pi + ambient_specular) * ao_texture.Sample(default_sampler, uv).x;


	shadow_map_uv /= shadow_map_uv.w;
	shadow_map_uv.y *= -1;
	float shadow_mask = saturate(map(length(shadow_map_uv.xyz), 0.9, 1, 0, 1));
	shadow_map_uv = shadow_map_uv * 0.5 + 0.5;

	float lightness = lerp(shadow_texture.SampleCmpLevelZero(shadow_sampler, shadow_map_uv.xy, shadow_map_uv.z).x, 1, shadow_mask);
	pixel_color.rgb = ambient + (diffuse + specular) * lightness;

	float fog = min(1, length(view) / (CHUNKW*FARD));
	fog *= fog;
	fog *= fog;
	pixel_color.rgb = lerp(pixel_color.rgb, ambient_color, fog);

	//pixel_color = shadowtex.SampleCmpLevelZero(dsam, shadow_map_uv.xy, shadow_map_uv.z-0.01);
	//pixel_color = shadowtex.Sample(sam, shadow_map_uv.xy);
}
)"s);

	tree_shadow_vs = create_vs(HLSL_CBUFFER R"(
struct Vertex {
	float3 position;
	float3 normal;
	float4 tangent;
	float4 color;
	float2 uv;
};

StructuredBuffer<Vertex> s_vertex_buffer : VERTEX_BUFFER_SLOT;

struct Instance {
	float3x3 mat;
	float3 position;
};

StructuredBuffer<Instance> s_instance_buffer : INSTANCE_BUFFER_SLOT;


void main(
	in uint vertex_id : SV_VertexID,
	in uint instance_id : SV_InstanceID,
	out float2 uv : UV,
	out float4 screen_uv : SCREEN_UV,
	out float4 position : SV_Position
) {
	Instance instance = s_instance_buffer[instance_id];
	float3 pos = mul(instance.mat, s_vertex_buffer[vertex_id].position) + instance.position + c_relative_position;
	position = mul(c_mvp, float4(pos, 1.0f));
	screen_uv = position * float4(1, -1, 1, 1);
	uv = s_vertex_buffer[vertex_id].uv;
}
)"s);

	tree_shadow_ps = create_ps(HLSL_CBUFFER HLSL_COMMON HLSL_LIGHTING HLSL_TRIPLANAR R"(
void main(
	in float2 uv : UV,
	in float4 screen_uv : SCREEN_UV
) {
	float4 data = albedo_texture.Sample(default_sampler, uv);
	clip(data.a - 0.5);
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
		D3D11_BLEND_DESC desc {
			.AlphaToCoverageEnable = true,
			.RenderTarget = {
				{
					.BlendEnable = false,
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
		dhr(device->CreateBlendState(&desc, &alpha_to_coverage_blend));
	}
	{
		D3D11_BLEND_DESC desc {
			.RenderTarget = {
				{
					.BlendEnable = true,
					.SrcBlend  = D3D11_BLEND_SRC1_COLOR,
					.DestBlend = D3D11_BLEND_INV_SRC1_COLOR,
					.BlendOp   = D3D11_BLEND_OP_ADD,
					.SrcBlendAlpha  = D3D11_BLEND_ZERO,
					.DestBlendAlpha = D3D11_BLEND_ZERO,
					.BlendOpAlpha   = D3D11_BLEND_OP_ADD,
					.RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL,
				}
			}
		};
		dhr(device->CreateBlendState(&desc, &font_blend));
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
		D3D11_RASTERIZER_DESC desc {
			.FillMode = D3D11_FILL_SOLID,
			.CullMode = D3D11_CULL_NONE,
		};
		dhr(device->CreateRasterizerState(&desc, &no_cull_rasterizer));
	}
	{
		D3D11_RASTERIZER_DESC desc {
			.FillMode = D3D11_FILL_SOLID,
			.CullMode = D3D11_CULL_BACK,
			.DepthBias = 1,
			.SlopeScaledDepthBias = 1,
		};
		dhr(device->CreateRasterizerState(&desc, &shadow_rasterizer));
	}
	{
		D3D11_RASTERIZER_DESC desc {
			.FillMode = D3D11_FILL_SOLID,
			.CullMode = D3D11_CULL_NONE,
			.DepthBias = 100,
			.SlopeScaledDepthBias = 1,
		};
		dhr(device->CreateRasterizerState(&desc, &no_cull_shadow_rasterizer));
	}

	{
		D3D11_SAMPLER_DESC desc {
			.Filter = D3D11_FILTER_ANISOTROPIC,
			.AddressU = D3D11_TEXTURE_ADDRESS_WRAP,
			.AddressV = D3D11_TEXTURE_ADDRESS_WRAP,
			.AddressW = D3D11_TEXTURE_ADDRESS_WRAP,
			.MaxAnisotropy = 16,
			.MinLOD = 0,
			.MaxLOD = max_value<f32>,
		};

		dhr(device->CreateSamplerState(&desc, &default_sampler_wrap));
	}
	{
		D3D11_SAMPLER_DESC desc {
			.Filter = D3D11_FILTER_ANISOTROPIC,
			.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP,
			.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP,
			.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP,
			.MaxAnisotropy = 16,
			.MinLOD = 0,
			.MaxLOD = max_value<f32>,
		};

		dhr(device->CreateSamplerState(&desc, &default_sampler_clamp));
	}
	{
		D3D11_SAMPLER_DESC desc {
			.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR,
			.AddressU = D3D11_TEXTURE_ADDRESS_WRAP,
			.AddressV = D3D11_TEXTURE_ADDRESS_WRAP,
			.AddressW = D3D11_TEXTURE_ADDRESS_WRAP,
			.MaxAnisotropy = 16,
			.MinLOD = 0,
			.MaxLOD = 0,
		};

		ID3D11SamplerState *sampler;
		dhr(device->CreateSamplerState(&desc, &sampler));

		immediate_context->PSSetSamplers(DEFAULT_NOMIP_SAMPLER_SLOT, 1, &sampler);
	}
	{
		D3D11_SAMPLER_DESC desc {
			.Filter = D3D11_FILTER_MIN_MAG_MIP_POINT,
			.AddressU = D3D11_TEXTURE_ADDRESS_WRAP,
			.AddressV = D3D11_TEXTURE_ADDRESS_WRAP,
			.AddressW = D3D11_TEXTURE_ADDRESS_WRAP,
			.MaxAnisotropy = 16,
			.MinLOD = 0,
			.MaxLOD = max_value<f32>,
		};

		ID3D11SamplerState *sampler;
		dhr(device->CreateSamplerState(&desc, &sampler));

		immediate_context->PSSetSamplers(NEAREST_SAMPLER_SLOT, 1, &sampler);
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

		immediate_context->PSSetSamplers(SHADOW_SAMPLER_SLOT, 1, &sampler);
	}

	{
		v4u8 pixels[256][256]{};
		for (s32 x = 0; x < 256; ++x) {
		for (s32 y = 0; y < 256; ++y) {
#if 0
			pixels[x][y] = (v4u8)V4f(255*pow2(1-length(V2f(x,y)-128)/(sqrtf(128*128*2))));
#else
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

				pixels[x][y] += (v4u8)V4u(sqrtf(min_distance_squared) * voronoi_inv_largest_possible_distance_2d*255 / 4);

				step_size *= 2;
			}
			pixels[x][y] = (v4u8)V4f(map_clamped<f32>(pixels[x][y].x/255.f, 0.1, 0.4, 0, 1)*255.f);
#endif
		}
		}

		voronoi_albedo = make_texture(pixels);

		v2f normalf[256][256]{};
		for (s32 x = 0; x < 256; ++x) {
		for (s32 y = 0; y < 256; ++y) {
			normalf[x][y] += V2f(
					pixels[x][y        ].x - pixels[(x+1)%256][y        ].x +
					pixels[x][(y+1)%256].x - pixels[(x+1)%256][(y+1)%256].x,
					pixels[x        ][y].x - pixels[x        ][(y+1)%256].x +
					pixels[(x+1)%256][y].x - pixels[(x+1)%256][(y+1)%256].x
			);
		}
		}

		v4u8 normal[256][256]{};
		for (s32 x = 0; x < 256; ++x) {
		for (s32 y = 0; y < 256; ++y) {
			auto &n = normal[y][x];
#if 0
			if (sqrtf(pow2(x-128)+pow2(y-128)) < 128) {
				n = {(u8)x,(u8)y,0,255};
			} else {
				n = {128,128,0,255};
			}

			n.z = (u8)map<f32>(1 - sqrtf(pow2(map<f32>(n.x, 0, 255, -1, 1)) + pow2(map<f32>(n.y, 0, 255, -1, 1))), 0, 1, 0, 255);
			n = {(u8)x,(u8)y,0,255};

			//normal[x][y] = {(u8)x,(u8)y,0,1};
#else
			if (length(normalf[y][x]) < 0.000001f) {
				normal[y][x] = {0,0,255,255};
			} else {
				auto nf = normalize(normalf[y][x]);
				normal[y][x] = (v4u8)map(V4f(
					nf.x,
					nf.y,
					sqrtf(nf.x*nf.x + nf.y*nf.y),
					1
				), V4f(-1,-1,0,0), V4f(1), V4f(0), V4f(255));
			}
#endif
		}
		}

		voronoi_normal = make_texture(normal);
	}

	{
		u8 pixels[16][16][16];
		u8 i = 0;
		for (auto &p : flatten(pixels)) {
			p = i++;
		}

		std::shuffle(flatten(pixels).begin(), flatten(pixels).end(), std::random_device{});

		lod_mask = make_texture(pixels, false);
		immediate_context->PSSetShaderResources(LOD_MASK_TEXTURE_SLOT, 1, &lod_mask);
	}

	{
		scoped_allocator(temporary_allocator);
		auto scene = parse_glb_from_memory(read_entire_file(format("{}/spruce.glb", executable_directory)));

		auto create_model = [&](Span<utf8> name) {
			auto mesh = find_if(scene.nodes, [&](auto &node) { return node.name == name; })->mesh;
			return Model {
				.vb = create_structured_buffer(mesh->vertices),
				.ib = create_index_buffer(mesh->indices),
				.index_count = (u32)mesh->indices.count,
			};
		};

		tree_model.add_lod(0, create_model(u8"lod0"s)).no_cull = true;
		tree_model.add_lod(1, create_model(u8"lod1"s)).no_cull = true;
		tree_model.add_lod(2, create_model(u8"lod2"s)).no_cull = true;
		tree_model.add_lod(3, create_model(u8"lod3"s)).no_cull = true;
		tree_model.add_lod(4, create_model(u8"lod4"s)).no_cull = true;
	}

	ID3D11ShaderResourceView *default_normal;
	{
		v4u8 normal[1][1] { 0x80, 0x80, 0xff, 0xff};
		default_normal = make_texture(normal);
	}

	ID3D11ShaderResourceView *default_ao;
	{
		v4u8 ao[1][1] { 0xff, 0xff, 0xff, 0xff};
		default_ao = make_texture(ao);
	}

	{
		int w,h;
		auto pixels = stbi_load(tformat("{}\\ground_albedo.png\0"s, executable_directory).data, &w, &h, 0, 4);
		defer { stbi_image_free(pixels); };
		ground_albedo = make_texture(pixels, w, h);
	}
	{
		int w,h;
		auto pixels = stbi_load(tformat("{}\\ground_normal.png\0"s, executable_directory).data, &w, &h, 0, 4);
		defer { stbi_image_free(pixels); };
		ground_normal = make_texture(pixels, w, h);
	}
	{
		int w,h;
		auto pixels = stbi_load(tformat("{}\\grass.png\0"s, executable_directory).data, &w, &h, 0, 4);
		defer { stbi_image_free(pixels); };
		grass_albedo = make_texture(pixels, w, h);
	}
	{
		int w,h;
		auto pixels = stbi_load(tformat("{}\\planks.png\0"s, executable_directory).data, &w, &h, 0, 4);
		defer { stbi_image_free(pixels); };
		planks_albedo = make_texture(pixels, w, h);
	}
	{
		int w,h;
		auto pixels = stbi_load(tformat("{}\\planks_normal.png\0"s, executable_directory).data, &w, &h, 0, 4);
		defer { stbi_image_free(pixels); };
		planks_normal = make_texture(pixels, w, h);
	}
	{
		int w,h;
		auto pixels = stbi_load(tformat("{}\\spruce_albedo.png\0"s, executable_directory).data, &w, &h, 0, 4);
		defer { stbi_image_free(pixels); };
		tree_model.lods[0].model.albedo =
		tree_model.lods[1].model.albedo =
		tree_model.lods[2].model.albedo = make_texture(pixels, w, h);
	}
	{
		// int w,h;
		// auto pixels = stbi_load(tformat("{}\\tree_normal.png\0"s, executable_directory).data, &w, &h, 0, 4);
		// defer { stbi_image_free(pixels); };
		// tree_normal = make_texture(pixels, w, h);
		tree_model.lods[0].model.normal =
		tree_model.lods[1].model.normal =
		tree_model.lods[2].model.normal = default_normal;
	}
	{
		int w,h;
		auto pixels = stbi_load(tformat("{}\\spruce_ao.png\0"s, executable_directory).data, &w, &h, 0, 4);
		defer { stbi_image_free(pixels); };
		tree_model.lods[0].model.ao =
		tree_model.lods[1].model.ao =
		tree_model.lods[2].model.ao = make_texture(pixels, w, h);
	}
	{
		int w,h;
		auto pixels = stbi_load(tformat("{}\\spruce_lod2_albedo.png\0"s, executable_directory).data, &w, &h, 0, 4);
		defer { stbi_image_free(pixels); };
		tree_model.lods[3].model.albedo =
		tree_model.lods[4].model.albedo = make_texture(pixels, w, h);
	}
	{
		int w,h;
		auto pixels = stbi_load(tformat("{}\\spruce_lod2_normal.png\0"s, executable_directory).data, &w, &h, 0, 4);
		defer { stbi_image_free(pixels); };
		tree_model.lods[3].model.normal =
		tree_model.lods[4].model.normal = default_normal;//make_texture(pixels, w, h);
	}
	{
		int w,h;
		auto pixels = stbi_load(tformat("{}\\spruce_lod2_ao.png\0"s, executable_directory).data, &w, &h, 0, 4);
		defer { stbi_image_free(pixels); };
		tree_model.lods[3].model.ao =
		tree_model.lods[4].model.ao = make_texture(pixels, w, h);
	}
	{
		ID3D11Texture2D *tex;
		defer { tex->Release(); };

		{
			D3D11_TEXTURE2D_DESC desc {
				.Width = shadow_map_size,
				.Height = shadow_map_size,
				.MipLevels = 1,
				.ArraySize = 1,
				.Format = DXGI_FORMAT_R32_TYPELESS,
				.SampleDesc = {1, 0},
				.Usage = D3D11_USAGE_DEFAULT,
				.BindFlags = D3D11_BIND_DEPTH_STENCIL|D3D11_BIND_SHADER_RESOURCE,
			};
			dhr(device->CreateTexture2D(&desc, 0, &tex));
		}
		{
			D3D11_DEPTH_STENCIL_VIEW_DESC desc {
				.Format = DXGI_FORMAT_D32_FLOAT,
				.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D,
				.Texture2D = {
					.MipSlice = 0
				},
			};
			dhr(device->CreateDepthStencilView(tex, &desc, &shadow_dsv));
		}
		{
			D3D11_SHADER_RESOURCE_VIEW_DESC desc {
				.Format = DXGI_FORMAT_R32_FLOAT,
				.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D,
				.Texture2D = {
					.MipLevels = 1
				},
			};
			dhr(device->CreateShaderResourceView(tex, &desc, &shadow_srv));
		}
	}



	font_collection = create_font_collection(as_span({
		(Span<utf8>)format(u8"{}\\segoeui.ttf"s, executable_directory),
	}));
	font_collection->update_atlas = [](TL_FONT_TEXTURE_HANDLE texture, void *data, v2u size) -> TL_FONT_TEXTURE_HANDLE {
		ID3D11Texture2D *buffer;
		defer { buffer->Release(); };

		List<v4u8> fourcomp;
		defer { free(fourcomp); };

		fourcomp.reserve(size.x*size.y);

		for (umm i = 0; i < size.x*size.y; ++i) {
			fourcomp.add((v4u8)V4u((v3u)((v3u8 *)data)[i], 0));
		}

		u32 pitch = sizeof(v4u8) * size.x;

		if (texture) {
			texture->GetResource((ID3D11Resource **)&buffer);

			immediate_context->UpdateSubresource(buffer, 0, 0, fourcomp.data, pitch, 1);
		} else {

			{
				D3D11_TEXTURE2D_DESC desc {
					.Width = size.x,
					.Height = size.y,
					.MipLevels = 1,
					.ArraySize = 1,
					.Format = DXGI_FORMAT_R8G8B8A8_UNORM,
					.SampleDesc = {1, 0},
					.BindFlags = D3D11_BIND_SHADER_RESOURCE,
				};

				D3D11_SUBRESOURCE_DATA init {
					.pSysMem = fourcomp.data,
					.SysMemPitch = pitch,
				};

				dhr(device->CreateTexture2D(&desc, &init, &buffer));
			}
			dhr(device->CreateShaderResourceView(buffer, 0, &texture));
		}
		return texture;
	};

	camera = &create_entity();
	camera_prev_position = camera->position = {0,-3,0};


	start();

	target_frame_time = 1 / 75.0f;

	lock_cursor();

	make_os_timing_precise();
	auto frame_time_counter = get_performance_counter();
	auto frame_timer = create_precise_timer();
	auto stat_reset_timer = get_performance_counter();
	the_timer = create_precise_timer();

    while (1) {
		profile_frame = key_held('T');
		if (profile_frame) {
			profiler.reset();
		}
		defer {
			if (profile_frame)
				write_entire_file("frame.tmd"s, (Span<u8>)profiler.output_for_timed());
			profile_frame = false;
		};

		timed_block(profiler, profile_frame, "frame");

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

		swap_chain->Present(1, 0);

		for (auto &state : key_state) {
			if (state & KeyState_up) {
				state = KeyState_none;
			}
			if (state & KeyState_down) {
				state &= ~KeyState_down;
			}
			if (state & KeyState_repeated) {
				state &= ~KeyState_repeated;
			}
		}

		// sync(frame_time_counter, target_frame_time);

		if (get_performance_counter() >= stat_reset_timer + performance_frequency) {
			stat_reset_timer += performance_frequency;
			generate_time.reset();
		}

		frame_time = reset(frame_timer);
		smooth_fps = lerp(smooth_fps, 1.0f / (f32)frame_time, 0.25f);

		clear_temporary_storage();

		++frame_number;
	}

    return 0;
}

void print_wsa_error() {
	switch (WSAGetLastError()) {
#define C(x) case x: print(#x); break;
		C(WSAEINTR                         )
		C(WSAEBADF                         )
		C(WSAEACCES                        )
		C(WSAEFAULT                        )
		C(WSAEINVAL                        )
		C(WSAEMFILE                        )
		C(WSAEWOULDBLOCK                   )
		C(WSAEINPROGRESS                   )
		C(WSAEALREADY                      )
		C(WSAENOTSOCK                      )
		C(WSAEDESTADDRREQ                  )
		C(WSAEMSGSIZE                      )
		C(WSAEPROTOTYPE                    )
		C(WSAENOPROTOOPT                   )
		C(WSAEPROTONOSUPPORT               )
		C(WSAESOCKTNOSUPPORT               )
		C(WSAEOPNOTSUPP                    )
		C(WSAEPFNOSUPPORT                  )
		C(WSAEAFNOSUPPORT                  )
		C(WSAEADDRINUSE                    )
		C(WSAEADDRNOTAVAIL                 )
		C(WSAENETDOWN                      )
		C(WSAENETUNREACH                   )
		C(WSAENETRESET                     )
		C(WSAECONNABORTED                  )
		C(WSAECONNRESET                    )
		C(WSAENOBUFS                       )
		C(WSAEISCONN                       )
		C(WSAENOTCONN                      )
		C(WSAESHUTDOWN                     )
		C(WSAETOOMANYREFS                  )
		C(WSAETIMEDOUT                     )
		C(WSAECONNREFUSED                  )
		C(WSAELOOP                         )
		C(WSAENAMETOOLONG                  )
		C(WSAEHOSTDOWN                     )
		C(WSAEHOSTUNREACH                  )
		C(WSAENOTEMPTY                     )
		C(WSAEPROCLIM                      )
		C(WSAEUSERS                        )
		C(WSAEDQUOT                        )
		C(WSAESTALE                        )
		C(WSAEREMOTE                       )
		C(WSASYSNOTREADY                   )
		C(WSAVERNOTSUPPORTED               )
		C(WSANOTINITIALISED                )
		C(WSAEDISCON                       )
		C(WSAENOMORE                       )
		C(WSAECANCELLED                    )
		C(WSAEINVALIDPROCTABLE             )
		C(WSAEINVALIDPROVIDER              )
		C(WSAEPROVIDERFAILEDINIT           )
		C(WSASYSCALLFAILURE                )
		C(WSASERVICE_NOT_FOUND             )
		C(WSATYPE_NOT_FOUND                )
		C(WSA_E_NO_MORE                    )
		C(WSA_E_CANCELLED                  )
		C(WSAEREFUSED                      )
		C(WSAHOST_NOT_FOUND                )
		C(WSATRY_AGAIN                     )
		C(WSANO_RECOVERY                   )
		C(WSANO_DATA                       )
		C(WSA_QOS_RECEIVERS                )
		C(WSA_QOS_SENDERS                  )
		C(WSA_QOS_NO_SENDERS               )
		C(WSA_QOS_NO_RECEIVERS             )
		C(WSA_QOS_REQUEST_CONFIRMED        )
		C(WSA_QOS_ADMISSION_FAILURE        )
		C(WSA_QOS_POLICY_FAILURE           )
		C(WSA_QOS_BAD_STYLE                )
		C(WSA_QOS_BAD_OBJECT               )
		C(WSA_QOS_TRAFFIC_CTRL_ERROR       )
		C(WSA_QOS_GENERIC_ERROR            )
		C(WSA_QOS_ESERVICETYPE             )
		C(WSA_QOS_EFLOWSPEC                )
		C(WSA_QOS_EPROVSPECBUF             )
		C(WSA_QOS_EFILTERSTYLE             )
		C(WSA_QOS_EFILTERTYPE              )
		C(WSA_QOS_EFILTERCOUNT             )
		C(WSA_QOS_EOBJLENGTH               )
		C(WSA_QOS_EFLOWCOUNT               )
		C(WSA_QOS_EUNKOWNPSOBJ             )
		C(WSA_QOS_EPOLICYOBJ               )
		C(WSA_QOS_EFLOWDESC                )
		C(WSA_QOS_EPSFLOWSPEC              )
		C(WSA_QOS_EPSFILTERSPEC            )
		C(WSA_QOS_ESDMODEOBJ               )
		C(WSA_QOS_ESHAPERATEOBJ            )
		C(WSA_QOS_RESERVED_PETYPE          )
		C(WSA_SECURE_HOST_NOT_FOUND        )
		C(WSA_IPSEC_NAME_POLICY_ERROR      )
		default: print("unknown WSA error"); break;
	}
	print("\n");
}
