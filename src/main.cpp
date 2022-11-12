// TODO:
// crash in tracking allocator.
//     when reallocating, the metainfo for old pointer is not found.

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
#include <tl/main.h>
#include <algorithm>
#include <random>

#pragma comment(lib, "freetype.lib")
#pragma comment(lib, "ws2_32.lib")

using namespace tl;

#include "shaders.h"
#include "input.h"

#include <freetype/freetype.h>
#define TL_FONT_TEXTURE_HANDLE ID3D11ShaderResourceView *
#include <tl/font.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#pragma push_macro("scoped_lock")
#undef scoped_lock
#include <reactphysics3d/reactphysics3d.h>
#pragma pop_macro("scoped_lock")
#pragma comment(lib, "reactphysics3d.lib")

// this is stupid
#pragma pop_macro("assert")

v4f random_v4f(v3s v) {
	return {
		random_f32(v),
		random_f32(v.yzx()),
		random_f32(v.zxy()),
		random_f32(v.zyx()),
	};
}

namespace p = reactphysics3d;

p::PhysicsCommon physics;
p::PhysicsWorld *physics_world;

v3f to_v3f(p::Vector3 v) { return {v.x, v.y, v.z}; }
p::Vector3 to_pv(v3f v) { return {v.x, v.y, v.z}; }

#define USE_INDICES 0

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
ID3D11RasterizerState *scissor_rasterizer;

bool wireframe_rasterizer_enabled;

ID3D11BlendState *alpha_blend;
ID3D11BlendState *font_blend;
ID3D11BlendState *alpha_to_coverage_blend;

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

ID3D11ShaderResourceView *default_albedo;
ID3D11ShaderResourceView *default_normal;
ID3D11ShaderResourceView *default_ao;

struct VertexBuffer {
	ID3D11ShaderResourceView *view = 0;
	u32 vertex_count = 0;
};

VertexBuffer farlands_vb;

List<utf8> executable_path;
Span<utf8> executable_directory;

u32 draw_mode;

u32 draw_call_count;

f32 frame_time;

u32 chunk_generation_amount_factor = 1;

bool draw_grass = true;
bool draw_trees = true;

bool draw_collision_shape = false;

struct BasicVertex {
	v3f position;
	v3f normal;
	v2f uv;
	v4f tangent;
};

struct Model {
	ID3D11ShaderResourceView *vb;
	ID3D11Buffer *ib;
	u32 index_count;

	ID3D11ShaderResourceView *albedo;
	ID3D11ShaderResourceView *normal;
	ID3D11ShaderResourceView *ao;
	bool no_cull = false;
};

struct LodModel {
	ID3D11ShaderResourceView *vertex_buffer;
	ID3D11Buffer *index_buffer;

	ID3D11ShaderResourceView *albedo;
	ID3D11ShaderResourceView *normal;
	ID3D11ShaderResourceView *ao;
	bool no_cull = false;

	struct Lod {
		u32 end_distance;
		bool cast_shadow;
		u32 index_count;
		u32 start_index;
		u32 start_vertex;
	};
	List<Lod> lods;
	Lod &get_lod(u32 distance) {
		for (auto &lod : lods) {
			if (distance <= lod.end_distance)
				return lod;
		}
		return lods.back();
	}
};

LodModel tree_model;
LodModel grenade_model;

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
bool show_profile;
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
		if (isnan(local.x) ||
			isnan(local.y) ||
			isnan(local.z))
		{
			with(ConsoleColor::red, print("local position is NAN!\n"));
			local = {};
			return;
		}
		auto chunk_offset = floor(floor_to_int(local), CHUNKW);
		chunk += chunk_offset / CHUNKW;
		local -= (v3f)chunk_offset;
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
	Entity *entity = 0;
	void start() {}
	void update() {}
	void on_free() {}
};

struct Entity {
	u32 entity_index;
	ChunkRelativePosition position = {};
	p::Quaternion rotation = p::Quaternion::identity();
	StaticList<ComponentId, 16> components;

	template <class T>
	T &add_component();

	template <class T>
	T *get_component();
};

StaticMaskedBlockList<Entity, 512> entities;

LinearSet<Entity *> entities_to_remove;

Entity *create_entity() {
	auto added = entities.add();
	added.pointer->entity_index = added.index;
	return added.pointer;
}

void enqueue_entity_removal(Entity *entity) {
	entities_to_remove.insert(entity);
}


template <class T>
using ComponentList = StaticMaskedBlockList<T, 256>;

template <class T>
ComponentList<T> components;

template <class T>
extern u64 component_index;

struct RemoverComponent : Component {
	f32 time_left;
	void update() {
		time_left -= target_frame_time;
		if (time_left < 0) {
			enqueue_entity_removal(entity);
		}
	}
};

struct PhysicsComponent : Component {
	p::RigidBody* body;
	p::Collider *collider;
	void start() {
		body = physics_world->createRigidBody({{}, p::Quaternion::identity()});
		body->setType(p::BodyType::DYNAMIC);
		collider = body->addCollider(physics.createBoxShape({1,1,1}), {});
	}
	void set_box_shape(aabb<v3f> box) {
		body->removeCollider(collider);
		collider = body->addCollider(physics.createBoxShape(to_pv(box.size() / 2)), {to_pv(box.center()), p::Quaternion::identity()});
		body->updateLocalCenterOfMassFromColliders();
	}
	void set_velocity(v3f velocity) {
		body->setLinearVelocity(to_pv(velocity));
	}
	void on_free() {
		physics_world->destroyRigidBody(body);
	}
};

struct ParticleSystemComponent : Component {
	struct Particle {
		v3f position;
		v3f velocity;
	};
	List<Particle> particles;

	v3s particles_chunk_position;
	void init(u32 particle_count) {
		particles_chunk_position = entity->position.chunk;
		particles.resize(particle_count);
		xorshift32 random { get_performance_counter() };
		for (auto &particle : particles) {
			particle.position = entity->position.local;
			particle.velocity = normalize(next_v3f(random) - 0.5f) * map<f32>(next_f32(random), 0, 1, 1, 2) * 10 * target_frame_time;
		}
	}
	void update() {
		for (auto &particle : particles) {
			particle.position += particle.velocity;
			particle.velocity += v3f{0,-9.8,0}*2*pow2(target_frame_time);
		}
	}
	void on_free() {
		free(particles);
	}
};

struct GrenadeComponent : Component {
	void on_free() {
		xorshift32 random { get_performance_counter() };

		for (u32 i = 0; i < 16; ++i) {
			add_sdf_sphere_synchronized(entity->position + (next_v3f(random) * 8 - 4), map<f32>(next_f32(random), 0, 1, 2, 4), -256);
		}

		enqueue_entity_removal(entity);

		auto ps_entity = create_entity();
		ps_entity->position = entity->position;

		auto &ps = ps_entity->add_component<ParticleSystemComponent>();
		ps.init(100);

		auto &remover = ps_entity->add_component<RemoverComponent>();
		remover.time_left = 1;
	}
};

struct MeshComponent : Component {
	LodModel *model;
};


/*
#define e(name)
ENUMERATE_COMPONENTS
#undef e
*/
#define ENUMERATE_COMPONENTS \
e(RemoverComponent) \
e(PhysicsComponent) \
e(ParticleSystemComponent) \
e(GrenadeComponent) \
e(MeshComponent) \

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

	component.pointer->start();

	return *component.pointer;
}

template <class T>
T *Entity::get_component() {
	for (auto component_index : components) {
		if (component_index.component_type == ::component_index<T>) {
			return &::components<T>[component_index.component_index];
		}
	}
	return 0;
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

v3s last_world_origin;
v3s current_world_origin;

v3f relative_to_origin(ChunkRelativePosition position) {
	position.chunk -= current_world_origin;
	return position.to_v3f();
}

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
	v3f parent_position;
	u32 parent_normal;
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

struct PhysicsVertex {
	v3f position;
	v3f normal;
};

struct Tree {
	v3f position;
	p::Quaternion rotation;
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

	List<PhysicsVertex> physics_vertices;
	List<u32> physics_indices;

	List<Tree> trees;

#if USE_INDICES
	List<u32> indices;
	u32 indices_count = 0;
#else
#endif

	Sdf *sdf = 0;

	Block (*blocks)[CHUNKW][CHUNKW][CHUNKW] = 0;

	p::RigidBody *body = 0;
	p::TriangleMesh *triangle_mesh = 0;
	p::TriangleVertexArray subpart;
	p::ConcaveMeshShape *shape = 0;

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

	RecursiveMutex mutex;

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

	v += floor(current_world_origin+DRAWD, s);

	v -= s * (v3s)(v > current_world_origin+DRAWD);

	auto r = absolute(v - current_world_origin);
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
	position += velocity*velocity_multiplier + acceleration*2*pow2(frame_time);


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
				if (sdf <= 0) {
					break;
				}
				if (auto gradient = gradient_at(corner_pos)) {
					corner_pos += gradient.value_unchecked() / iter_max;
				} else {
					corner_pos += v3f{0,1,0};
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

	for (s32 x = current_world_origin.x-DRAWD; x <= current_world_origin.x+DRAWD; ++x) {
	for (s32 y = current_world_origin.y-DRAWD; y <= current_world_origin.y+DRAWD; ++y) {
	for (s32 z = current_world_origin.z-DRAWD; z <= current_world_origin.z+DRAWD; ++z) {
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

	auto r = position != current_world_origin + DRAWD;
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
			Vertex _0 = {.position = position + V3f(m * v2f{-1,0}, -0.5f).xzy(), .normal = normal, .uv = {u+0.0f,1}};
			Vertex _1 = {.position = position + V3f(m * v2f{-1,0},h-0.5f).xzy(), .normal = normal, .uv = {u+0.0f,0}};
			Vertex _2 = {.position = position + V3f(m * v2f{+1,0}, -0.5f).xzy(), .normal = normal, .uv = {u+0.5f,1}};
			Vertex _3 = {.position = position + V3f(m * v2f{+1,0},h-0.5f).xzy(), .normal = normal, .uv = {u+0.5f,0}};
			_0.origin = _1.origin = _0.position;
			_2.origin = _3.origin = _2.position;
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

	chunk->trees.clear();

	auto put_tree = [&] (v3f position) {
		if (next_f32(rng) < 0.02f) {
			f32 angle = next_f32(rng)*tau;
			chunk->trees.add({position, p::Quaternion::fromEulerAngles(0,angle,0)});
			trees.add({
				.matrix = m3::rotation_r_y(angle) * m3::scale(map<f32>(next_f32(rng), 0, 1, 0.75, 1.25)),
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
void sdf_to_triangles(v3s lbounds, auto &&sdf, auto &&add_vertex, auto &&add_index)
	requires requires { {add_vertex(ChunkVertex{})}; {add_index(u32{})}; }
{

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

	// NOTE:
	// If we use 0 and 1 for corners and sdf happens to be is 0, then a vertex will be exactly on the boundary, which can cause multiple
	// vertices exist at the same position, which in turn will cause physics problems.
	// To prevent this, offset the corners by small enough value.
	f32 const e = 0.001;
	v3f corners[8] {
		{e,   e,   e  },
		{e,   e,   1-e},
		{e,   1-e, e  },
		{e,   1-e, 1-e},
		{1-e, e,   e  },
		{1-e, e,   1-e},
		{1-e, 1-e, e  },
		{1-e, 1-e, 1-e},
	};

	u32 indices[dim+3][dim+3][dim+3];

	u32 current_index = 0;
	u32 vertex_count = 0;

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

				ChunkVertex vertex;
				vertex.position = point + V3f(lx,ly,lz);

				// NOTE: Maybe there is no need to convert this to float and back to int?
				vertex.normal = encode_normal(normalize((v3f)v3s{
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

				indices[lx][ly][lz] = current_index++;

				add_vertex(vertex);
				++vertex_count;
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
			auto quad = [&](v3s _0, v3s _1, v3s _2, v3s _3) {
				auto i0 = indices[_0.x][_0.y][_0.z];
				auto i1 = indices[_1.x][_1.y][_1.z];
				auto i2 = indices[_2.x][_2.y][_2.z];
				auto i3 = indices[_3.x][_3.y][_3.z];

				assert(i0 < vertex_count);
				assert(i1 < vertex_count);
				assert(i2 < vertex_count);
				assert(i3 < vertex_count);

				add_index(i0);
				add_index(i1);
				add_index(i2);
				add_index(i1);
				add_index(i3);
				add_index(i2);
			};

			auto s0 = sdf(lx,ly,lz);
			auto s1 = sdf(lx+1,ly,lz);
			auto s2 = sdf(lx,ly+1,lz);
			auto s3 = sdf(lx,ly,lz+1);
			if ((s0 > 0) != (s1 > 0)) {
				v3s _0 = {lx, ly-1, lz-1};
				v3s _1 = {lx, ly+0, lz-1};
				v3s _2 = {lx, ly-1, lz+0};
				v3s _3 = {lx, ly+0, lz+0};

				if (!(s0 > 0)) {
					swap(_0, _3);
				}

				quad(_0, _1, _2, _3);
			}
			if ((s0 > 0) != (s2 > 0)) {
				v3s _0 = {lx-1, ly, lz-1};
				v3s _1 = {lx+0, ly, lz-1};
				v3s _2 = {lx-1, ly, lz+0};
				v3s _3 = {lx+0, ly, lz+0};

				if (s0 > 0) {
					swap(_0, _3);
				}

				quad(_0, _1, _2, _3);
			}
			if ((s0 > 0) != (s3 > 0)) {
				v3s _0 = {lx-1, ly-1, lz};
				v3s _1 = {lx+0, ly-1, lz};
				v3s _2 = {lx-1, ly+0, lz};
				v3s _3 = {lx+0, ly+0, lz};

				if (!(s0 > 0)) {
					swap(_0, _3);
				}

				quad(_0, _1, _2, _3);
			}
			int cpp_debugging_is_great = 42;
		}
		}
		}
	}
}

using ChunkVertexArena = StaticList<ChunkVertex, 1024*1024>;

template <u32 lodw>
void generate_chunk_lod(Chunk *chunk, v3s chunk_position, ChunkVertexArena &vertices, Span<ChunkVertex> parent_vertices) {
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

	StaticList<ChunkVertex, 1024*1024> unique_vertices;
	StaticList<u32, 1024*1024> indices;

	sdf_to_triangles<lodw>(lbounds, sdf,
		[&](ChunkVertex vertex) {
			vertex.position *= CHUNKW/lodw;
			vertex.position += 0.5f * (1 << (log2(CHUNKW) - log2(lodw))) - 0.5f;
			unique_vertices.add(vertex);
		},
		[&](u32 i) {
			vertices.add(unique_vertices[i]);
			indices.add(i);
		}
	);

#if 0
	for (auto &vertex : vertices) {
		f32 mind = max_value<f32>;
		ChunkVertex closest = vertex;
		for (auto &parent_vertex : parent_vertices) {
			f32 d = distance(vertex.position, parent_vertex.position);
			if (mind > d) {
				mind = d;
				closest = parent_vertex;
			}
		}
		vertex.parent_position = closest.position;
		vertex.parent_normal = closest.normal;
	}
#endif

	auto vertex_count = vertices.count - start_vertex;

	auto lod_index = log2(CHUNKW) - log2(lodw);

	chunk->sdf_vb_vertex_offset[lod_index] = start_vertex;
	chunk->sdf_vb_vertex_count [lod_index] = vertex_count;

	// NOTE: physics are really slow with lod0
	if (lod_index == 1) {
		chunk->physics_vertices.clear();
		chunk->physics_indices.clear();
		for (auto &v : unique_vertices) {
			chunk->physics_vertices.add({v.position, decode_normal(v.normal)});
		}
		for (auto i : indices) {
			chunk->physics_indices.add(i);
		}
	}

	chunk->frames_since_remesh = 0;
}

void update_chunk_mesh(Chunk *chunk, v3s chunk_position) {
	timed_function(profiler, profile_frame);

	u32 locked_by;
	if (!try_lock(chunk->mutex, &locked_by)) {
		with(ConsoleColor::red, print("Attempt to access the same chunk from different threads: {} and {}\n", locked_by, get_current_thread_id()));
		invalid_code_path("shared chunk access must not happen!");
	}
	defer { unlock(chunk->mutex); };

	ChunkVertexArena vertices;
	Span<ChunkVertex> prev_lod;

	auto do_lod = [&]<umm lodw>() {
		auto start = vertices.end();
		generate_chunk_lod<lodw>(chunk, chunk_position, vertices, prev_lod);
		prev_lod = {start, vertices.end()};
	};

	do_lod.operator()<1 >();
	do_lod.operator()<2 >();
	do_lod.operator()<4 >();
	do_lod.operator()<8 >();
	do_lod.operator()<16>();
	do_lod.operator()<32>();


	if (vertices.count) {
		chunk->sdf_vb = create_structured_buffer(vertices.span());
	}

	chunk->sdf_mesh_generated = true;
}

void start() {

}

bool chunk_in_view(FrustumPlanes frustum, v3s position, f32 radius_scale = 1.0f) {
	auto relative_position = (v3f)((position - current_world_origin)*CHUNKW);

	f32 const radius = CHUNKW * sqrt3 * 0.5f * radius_scale;
	return contains_sphere(frustum, relative_position+V3f(CHUNKW/2), radius);
}

extern "C" const Array<v3s8, pow3(DRAWD*2+1)> grid_map;

u32 thread_count;

u32 first_not_generated_sdf_chunk_index = 0;
u32 first_not_fully_meshed_chunk_index = 0;
u32 fully_meshed_chunk_index_end = 0;

s32 n_sdfs_can_generate_this_frame;
u32 remesh_count = 0;

f32 get_chunk_lod_t(v3s p) {
	//auto distance = max(absolute(p - camera->position.chunk));
	auto distance = length((v3f)(p - camera->position.chunk) + 0.5f - camera->position.local/CHUNKW);
	return log2(max(distance, 1));
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

void free_physics(Chunk *chunk) {
	if (chunk->body) {
		physics_world->destroyRigidBody(chunk->body);
		chunk->body = 0;

		physics.destroyTriangleMesh(chunk->triangle_mesh);
		chunk->triangle_mesh = 0;

		physics.destroyConcaveMeshShape(chunk->shape);
		chunk->shape = 0;
	}
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
	// Dude what? How can this fail.
	// assert(update_time_except_gen_time >= 0);

	f32 available_time = max(0, target_frame_time - update_time_except_gen_time);

	n_sdfs_can_generate_this_frame =
		smoothed_average_generation_time == 0 ?
		thread_count :
		chunk_generation_amount_factor * thread_count * max(1, floor_to_int(available_time / smoothed_average_generation_time));

	auto work = make_work_queue(thread_pool);

	if (any_true(current_world_origin != last_world_origin)) {
		timed_block(profiler, profile_frame, "remove chunks");
		first_not_generated_sdf_chunk_index = 0;
		first_not_fully_meshed_chunk_index = 0;

		for (s32 x = last_world_origin.x-DRAWD; x <= last_world_origin.x+DRAWD; ++x) {
		for (s32 y = last_world_origin.y-DRAWD; y <= last_world_origin.y+DRAWD; ++y) {
		for (s32 z = last_world_origin.z-DRAWD; z <= last_world_origin.z+DRAWD; ++z) {
			if (current_world_origin.x-DRAWD <= x && x <= current_world_origin.x+DRAWD)
			if (current_world_origin.y-DRAWD <= y && y <= current_world_origin.y+DRAWD)
			if (current_world_origin.z-DRAWD <= z && z <= current_world_origin.z+DRAWD)
				continue;

			auto chunk_position = v3s{x,y,z};

			auto r = absolute(chunk_position - current_world_origin);
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
			chunk->sdf_mesh_generated = false;

			free(chunk->trees_instances_buffer);
			chunk->trees.clear();

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

			// NOTE: keep allocated buffer.
			// free(chunk->physics_vertices);
			// free(chunk->physics_indices);

			chunk->lod_t = 1;

			free_physics(chunk);
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
				v3s p = v3s{x,y,z} + current_world_origin - dim/2;

				auto d = f32x8_clamp(sdf_func_x8(p*CHUNKW, CHUNKW), f32x8_set1(-128), f32x8_set1(127));
				for (s32 i = 0; i < 8; ++i) {
					v3s l = {p.x,p.y,p.z+i};
					if (all_true(current_world_origin - DRAWD <= l && l <= current_world_origin + DRAWD) && get_chunk(l)->sdf_mesh_generated) {
						sdf[x][y][z+i] = -128;
					} else {
						if (auto found = saved_chunks.find(l)) {
							sdf[x][y][z+i] = found->value.sdf[0][0][0];
						} else {
							sdf[x][y][z+i] = (s8)d.m256_f32[i];
						}
					}
				}
			}
			}
			}

			StaticList<ChunkVertex, 1024*1024> unique_vertices;

			sdf_to_triangles<dim>(V3s(dim),
				[&](s32 x, s32 y, s32 z) {
					return sdf[x][y][z];
				},
				[&](ChunkVertex vertex) {
					vertex.position = (vertex.position - dim/2) * CHUNKW;
					//point += 8;
					unique_vertices.add(vertex);
				},
				[&](u32 i) {
					vertices.add(unique_vertices[i]);
				}
			);

			farlands_vb.vertex_count = vertices.count;
			if (vertices.count) {
				farlands_vb.view = create_structured_buffer(vertices.span());
			}
		}

	}

	List<Chunk *> chunks_that_need_collision_update;
	chunks_that_need_collision_update.allocator = temporary_allocator;

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
			auto chunk_position = current_world_origin + p;
			auto chunk = get_chunk(chunk_position);
			if (chunk->sdf_generated) {
				if (chunk->needs_filter_and_remesh) {
					// Chunk was changed by somebody
					chunk->needs_filter_and_remesh = false;
					work.push([chunk = chunk, chunk_position] {
						filter_sdf(chunk);
						update_chunk_mesh(chunk, chunk_position);
						update_chunk_mask(chunk, chunk_position);
					});
					remesh_count++;
					free_physics(chunk);
					chunks_that_need_collision_update.add(chunk);
				} else {
					if (remesh_count < n_sdfs_can_generate_this_frame) {
						auto new_neighbor_mask = get_neighbor_mask(chunk_position);
						auto r = chunk_position == current_world_origin + DRAWD;
						if (chunk->neighbor_mask != new_neighbor_mask) {
							if ((new_neighbor_mask.x || r.x) && (new_neighbor_mask.y || r.y) && (new_neighbor_mask.z || r.z)) {
								// All neighbors are generated and full mesh can be created
								chunk->neighbor_mask = new_neighbor_mask;
								work.push([chunk, chunk_position] {
									update_chunk_mesh(chunk, chunk_position);
									update_chunk_mask(chunk, chunk_position);
								});
								remesh_count++;
								free_physics(chunk);
								chunks_that_need_collision_update.add(chunk);
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

					chunk->lod_previous_frame = (u32)get_chunk_lod_t(chunk_position);

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
				auto e = chunk_position == current_world_origin + DRAWD;
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
			auto chunk_position = current_world_origin + p;

			auto lod_index = (u32)get_chunk_lod_t(p);

			//if (visible)
			//	if (lodw == 32)
			//		debug_break();

			bool did_remesh = false;

			auto &chunk = get_chunk(chunk_position);

			if (chunk.sdf_generated) {
			}

			bool has_full_mesh =
				any_true(chunk_position == current_world_origin + DRAWD) ?
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

	for (auto chunk : chunks_that_need_collision_update) {
		if (chunk->physics_indices.count) {
			auto chunk_position = get_chunk_position(chunk);
			chunk->body = physics_world->createRigidBody({{}, p::Quaternion::identity()});
			chunk->body->setType(p::BodyType::STATIC);
			chunk->triangle_mesh = physics.createTriangleMesh();
#if 1
			new (&chunk->subpart) p::TriangleVertexArray(
				chunk->physics_vertices.count,
				&chunk->physics_vertices[0].position, sizeof(chunk->physics_vertices[0]),
				chunk->physics_indices.count/3, &chunk->physics_indices[0], sizeof(chunk->physics_indices[0])*3,
				p::TriangleVertexArray::VertexDataType::VERTEX_FLOAT_TYPE,
				p::TriangleVertexArray::IndexDataType::INDEX_INTEGER_TYPE
			);
#else
			new (&chunk->subpart) p::TriangleVertexArray(
				chunk->physics_vertices.count,
				&chunk->physics_vertices[0].position, sizeof(chunk->physics_vertices[0]),
				&chunk->physics_vertices[0].normal, sizeof(chunk->physics_vertices[0]),
				chunk->physics_indices.count/3, &chunk->physics_indices[0], sizeof(chunk->physics_indices[0])*3,
				p::TriangleVertexArray::VertexDataType::VERTEX_FLOAT_TYPE,
				p::TriangleVertexArray::NormalDataType::NORMAL_FLOAT_TYPE,
				p::TriangleVertexArray::IndexDataType::INDEX_INTEGER_TYPE
			);
#endif
			chunk->triangle_mesh->addSubpart(&chunk->subpart);
			chunk->shape = physics.createConcaveMeshShape(chunk->triangle_mesh);
			chunk->body->addCollider(chunk->shape, {});

			aabb<v3f> box {
				{-1,  0, -1},
				{ 1, 20,  1},
			};
			for (auto &tree : chunk->trees) {
				chunk->body->addCollider(physics.createBoxShape(to_pv(box.size() / 2)), {to_pv(tree.position + box.center()), p::Quaternion::identity()});
			}
		}
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
			for (u32 i = 0; i < chunk->physics_indices.count; i += 3) {
				auto a = chunk->physics_vertices[chunk->physics_indices[i+0]].position;
				auto b = chunk->physics_vertices[chunk->physics_indices[i+1]].position;
				auto c = chunk->physics_vertices[chunk->physics_indices[i+2]].position;
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

			for (auto &tree : chunk->trees) {
				auto tree_entity = create_entity();
				tree_entity->position.chunk = {x,y,z};
				tree_entity->position.local = tree.position;

				tree_entity->rotation = tree.rotation;

				auto &physics = tree_entity->add_component<PhysicsComponent>();
				physics.set_box_shape({{-1,0,-1},{1,20,1}});

				auto &mesh = tree_entity->add_component<MeshComponent>();
				mesh.model = &tree_model;

				auto &remover = tree_entity->add_component<RemoverComponent>();
				remover.time_left = 10;
			}
			chunk->trees.clear();

			free(chunk->trees_instances_buffer);
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

void draw_vertices(u32 vertex_count, u32 start_vertex = 0) {
	immediate_context->Draw(vertex_count, start_vertex);
	++draw_call_count;
}

void draw_indexed(u32 index_count, u32 start_index = 0, u32 start_vertex = 0) {
	immediate_context->DrawIndexed(index_count, start_index, start_vertex);
	++draw_call_count;
}

void draw_text(Span<utf8> str, u32 size, v2f position, bool ndc_position = false) {
	auto font = get_font_at_size(font_collection, size);
	ensure_all_chars_present(str, font);
	auto placed_chars = place_text(str, font);
	defer { free(placed_chars); };

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

		if (font_vertex_buffer) {
			font_vertex_buffer->Release();
			font_vertex_buffer = 0;
		}
		font_vertex_buffer = create_structured_buffer(vertices.span());

		immediate_context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
		immediate_context->VSSetShaderResources(0, 1, &font_vertex_buffer);
		immediate_context->PSSetShaderResources(0, 1, &font->texture);
		immediate_context->VSSetShader(shaders.font_vs, 0, 0);
		immediate_context->PSSetShader(shaders.font_ps, 0, 0);
		immediate_context->OMSetBlendState(font_blend, {}, -1);

		draw_vertices(vertices.count);
	}
}

List<utf8> chunk_info_string;
List<utf8> allocation_info_string;
PreciseTimer the_timer;
u32 frame_number;

List<Profiler::TimeSpan> previous_frame_profile;

struct GpuTimeQuery {
	ID3D11Query *disjoint;
	ID3D11Query *start;
	ID3D11Query *end;
	char const *name;
};

ID3D11Query *create_query(D3D11_QUERY kind) {
	D3D11_QUERY_DESC desc {
		.Query = kind,
	};
	ID3D11Query *query;
	dhr(device->CreateQuery(&desc, &query));
	return query;
}

GpuTimeQuery start_gpu_timer() {
	GpuTimeQuery query = {};
	query.disjoint = create_query(D3D11_QUERY_TIMESTAMP_DISJOINT);
	query.start    = create_query(D3D11_QUERY_TIMESTAMP);
	query.end      = create_query(D3D11_QUERY_TIMESTAMP);

	immediate_context->Begin(query.disjoint);
	immediate_context->End(query.start);
	return query;
}

void stop_gpu_timer(GpuTimeQuery query) {
	immediate_context->End(query.end);
	immediate_context->End(query.disjoint);
}

struct GpuTime {
	u64 start;
	u64 end;
	u64 frequency;
};

GpuTime get_gpu_time(GpuTimeQuery query) {
    UINT64 start_time = 0;
    while(immediate_context->GetData(query.start, &start_time, sizeof(start_time), 0) != S_OK);

    UINT64 end_time = 0;
    while(immediate_context->GetData(query.end, &end_time, sizeof(end_time), 0) != S_OK);

    D3D11_QUERY_DATA_TIMESTAMP_DISJOINT disjoint_data;
    while(immediate_context->GetData(query.disjoint, &disjoint_data, sizeof(disjoint_data), 0) != S_OK);

	if (disjoint_data.Disjoint)
		return {};

	return {
		.start = start_time,
		.end = end_time,
		.frequency = disjoint_data.Frequency,
	};
}

List<GpuTimeQuery> gpu_time_queries;

void update() {
	is_connected = another_thread_is_connected;

	v3s new_world_origin = current_world_origin;
	defer {
		last_world_origin = current_world_origin;
		current_world_origin = new_world_origin;
	};

	new_world_origin = camera->position.chunk;

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
		if (chunk_generation_amount_factor == 1)
			chunk_generation_amount_factor = 0;
		else if (chunk_generation_amount_factor != 1)
			chunk_generation_amount_factor /= 2;
	}
	if (key_down(Key_plus)) {
		if (chunk_generation_amount_factor == 0)
			chunk_generation_amount_factor = 1;
		else if (chunk_generation_amount_factor != 64)
			chunk_generation_amount_factor *= 2;
	}

	if (key_down(Key_left_bracket )) draw_grass = !draw_grass;
	if (key_down(Key_right_bracket)) draw_trees = !draw_trees;

	if (key_down(Key_semicolon)) {
		draw_collision_shape = !draw_collision_shape;
		physics_world->getDebugRenderer().setIsDebugItemDisplayed(p::DebugRenderer::DebugItem::COLLISION_SHAPE, draw_collision_shape);
	}

	if (key_down(Key_comma)) {
		draw_mode = (draw_mode == 0 ? DRAW_MODE_COUNT : draw_mode) - 1;
	}
	if (key_down(Key_period)) {
		draw_mode = (draw_mode == DRAW_MODE_COUNT-1 ? 0 : draw_mode+1);
	}

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
		camera->position = {0,1,0};
	}
	if (key_down('J')) {
		camera->position.chunk.x *= 2;
	}

	{
		switch (camera_mode) {
			case CameraMode::fly: {
				camera->position += camera_rotation * camera_position_delta * frame_time * 32 * (key_held(Key_shift) ? 10 : 1);

				if (key_down('F')) {
					camera_mode = CameraMode::walk;
					camera->get_component<PhysicsComponent>()->body->setIsActive(true);
				}
				break;
			}
			case CameraMode::walk: {
				/*
				auto xz_delta = m4::rotation_r_y(camera_angles.y) * camera_position_delta * v3f{1,0,1} * frame_time * (key_held(Key_shift) ? 2 : 1);
				camera->position += xz_delta;
				if (key_down(' '))
					camera->position += v3f{0,frame_time*7,0};

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

				*/

				auto ph = camera->get_component<PhysicsComponent>();

				auto xz_delta = m4::rotation_r_y(camera_angles.y) * camera_position_delta * v3f{1,0,1} * 100 * (key_held(Key_shift) ? 2 : 1);
				ph->body->applyWorldForceAtCenterOfMass(to_pv(xz_delta));
				auto vel = ph->body->getLinearVelocity();
				vel.x *= 0.5f;
				vel.z *= 0.5f;
				ph->body->setLinearVelocity(vel);


				if (key_down(' '))
					ph->body->applyWorldForceAtCenterOfMass({0,700,0});

				if (key_down('F')) {
					ph->body->setIsActive(false);
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

	if (mouse_down(2)) {
		auto grenade_entity = create_entity();
		grenade_entity->position = camera->position + camera_forward;

		auto &grenade = grenade_entity->add_component<GrenadeComponent>();

		auto &remover = grenade_entity->add_component<RemoverComponent>();
		remover.time_left = 3;

		auto &physics = grenade_entity->add_component<PhysicsComponent>();
		physics.set_velocity(camera_forward*50);

		auto &mesh = grenade_entity->add_component<MeshComponent>();
		mesh.model = &grenade_model;
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
physics_vertices.count: {}
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
chunk->physics_vertices.count,
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

	free(allocation_info_string);
	if (update_allocations) {
		StringBuilder builder;
		defer { free(builder); };

		append(builder, "location: current | total\n");
		for_each(tracked_allocations, [&](auto location, auto info) {
			append_format(builder, "{}: {} | {}\n", location, format_bytes(info.current_size), format_bytes(info.total_size));
		});

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


	//
	// REMOVE QUEUED ENTITIES
	//

	for (auto entity : entities_to_remove) {
		for (auto &component_index : entity->components) {
			switch (component_index.component_type) {
#define e(name) \
	case name##_index: \
		components<name>.at(component_index.component_index).on_free(); \
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
	// PHYSICS
	//
	for (u64 mask_index = 0; mask_index != count_of(nonempty_chunk_mask); ++mask_index) {
		auto mask = nonempty_chunk_mask[mask_index];
		if (mask == 0)
			continue;

		for (u64 bit_index = 0; bit_index != 64; ++bit_index) {
			if (mask & ((u64)1 << bit_index)) {
				auto chunk_position = get_chunk_position(mask_index * bits_in_chunk_mask + bit_index);
				auto chunk = get_chunk(chunk_position);
				auto relative_position = (v3f)((chunk_position-current_world_origin)*CHUNKW);

				if (chunk->body) {
					chunk->body->setTransform({to_pv(relative_position), p::Quaternion::identity()});
					//chunk->body->setTransform({to_pv((v3f)chunk_position * CHUNKW), p::Quaternion::identity()});
				}
			}
		}
	}

	for (auto &comp : components<PhysicsComponent>) {
		auto relative_position = comp.entity->position;
		relative_position.chunk -= current_world_origin;

		comp.body->setTransform({
				to_pv(relative_position.to_v3f()),
				comp.entity->rotation
			},
			false
		);
	}

	static f32 physics_time = 0;
	physics_time += frame_time;
	for (u32 iters = 0; iters < 4 && physics_time >= target_frame_time; ++iters) {
		physics_time -= target_frame_time;
		physics_world->update(target_frame_time);
	}
	for (auto &comp : components<PhysicsComponent>) {
		comp.entity->position.chunk = current_world_origin;
		comp.entity->position.local = to_v3f(comp.body->getTransform().getPosition());
		comp.entity->position.normalize();

		comp.entity->rotation = comp.body->getTransform().getOrientation();
	}


	//
	// GRAPHICS
	//

	draw_call_count = 0;
	if (profile_frame)
		gpu_time_queries.clear();

	immediate_context->VSSetConstantBuffers(0, 1, &frame_cbuffer.cbuffer);
	immediate_context->PSSetConstantBuffers(0, 1, &frame_cbuffer.cbuffer);
	immediate_context->VSSetConstantBuffers(1, 1, &entity_cbuffer.cbuffer);
	immediate_context->PSSetConstantBuffers(1, 1, &entity_cbuffer.cbuffer);
	immediate_context->PSSetSamplers(DEFAULT_SAMPLER_SLOT, 1, &default_sampler_wrap);

	StaticList<Chunk *, pow3(DRAWD*2+1)> visible_chunks;

	u32 num_masks_skipped = 0;

#define scoped_gpu_timer(_name) \
	timed_block(profiler, profile_frame, _name); \
	GpuTimeQuery gpu_timer; \
	if (profile_frame) { \
		gpu_timer = start_gpu_timer(); \
		gpu_timer.name = _name; \
	} \
	defer { \
		if (profile_frame) { \
			stop_gpu_timer(gpu_timer); \
			gpu_time_queries.add(gpu_timer); \
		} \
	};

	auto camera_position_relative_to_origin = relative_to_origin(camera->position);

	auto rotproj = m4::perspective_left_handed((f32)window_client_size.x / window_client_size.y, radians(camera_fov), 0.1, CHUNKW * FARD) * m4::rotation_r_yxz(-camera_angles);
	auto camera_matrix = rotproj * m4::translation(-camera_position_relative_to_origin);

	auto camera_frustum = create_frustum_planes_d3d(camera_matrix);

	//
	// SHADOWS
	//
	{
		scoped_gpu_timer("shadows");

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
		f32 const snap_size = shadow_pixels_in_meter;

		auto lightr = m4::rotation_r_yxz(-light_angles);

		v3f lightpos = camera->position.to_v3f();
		lightpos = lightr * lightpos;
		lightpos *= snap_size;
		lightpos = round(lightpos);
		lightpos /= snap_size;
		lightpos = inverse(lightr) * lightpos;
		lightpos -= (v3f)(current_world_origin * CHUNKW);

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
		immediate_context->VSSetShader(shaders.shadow_vs, 0, 0);
		immediate_context->PSSetShader(shaders.shadow_ps, 0, 0);
		immediate_context->RSSetState(shadow_rasterizer);
		ID3D11ShaderResourceView *null = 0;
		immediate_context->PSSetShaderResources(SHADOW_TEXTURE_SLOT, 1, &null);
		immediate_context->OMSetRenderTargets(0, 0, shadow_dsv);
		immediate_context->OMSetBlendState(0, {}, -1);
		immediate_context->ClearDepthStencilView(shadow_dsv, D3D11_CLEAR_DEPTH, 1, 0);

		//{
		//	timed_block(profiler, profile_frame, "far lands shadow");
		//
		//	if (farlands_vb.vertex_count) {
		//		entity_cbuffer.update({
		//			.relative_position = {},
		//			.vertex_offset = 0,
		//		});
		//
		//		immediate_context->VSSetShaderResources(VERTEX_BUFFER_SLOT, 1, &farlands_vb.view);
		//		draw_vertices(farlands_vb.vertex_count);
		//	}
		//	//immediate_context->ClearDepthStencilView(shadow_dsv, D3D11_CLEAR_DEPTH, 1, 0);
		//}

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
				if (chunk_in_view(light_frustum, chunk_position)) {
					visible_chunks.add(chunk);

					f32 lod_t = get_chunk_lod_t(chunk_position);
					auto lod_index = debug_selected_lod == -1 ? (u32)lod_t : debug_selected_lod;

					if (chunk->sdf_vb_vertex_count[lod_index]) {
						auto relative_position = (v3f)(chunk_position - current_world_origin) * CHUNKW;
						entity_cbuffer.update({
							.relative_position = relative_position,
							.vertex_offset = chunk->sdf_vb_vertex_offset[lod_index],
							.lod_t = lod_t - lod_index,
						});

						immediate_context->VSSetShaderResources(VERTEX_BUFFER_SLOT, 1, &chunk->sdf_vb);
						draw_vertices(chunk->sdf_vb_vertex_count[lod_index]);

						if (chunk->blocks_vb.view) {
							immediate_context->VSSetShaderResources(VERTEX_BUFFER_SLOT, 1, &chunk->blocks_vb.view);
							draw_vertices(chunk->blocks_vb.vertex_count);
						}
					}
				}
					}}}
		}



		if (draw_trees) {
			timed_block(profiler, profile_frame, "tree shadow");
			immediate_context->RSSetState(no_cull_shadow_rasterizer);
			immediate_context->VSSetShader(shaders.tree_shadow_vs, 0, 0);
			immediate_context->PSSetShader(shaders.tree_shadow_ps, 0, 0);

			immediate_context->VSSetShaderResources(VERTEX_BUFFER_SLOT, 1, &tree_model.vertex_buffer);
			immediate_context->PSSetShaderResources(ALBEDO_TEXTURE_SLOT, 1, &tree_model.albedo);
			immediate_context->IASetIndexBuffer(tree_model.index_buffer, DXGI_FORMAT_R32_UINT, 0);
			immediate_context->RSSetState(tree_model.no_cull ? no_cull_shadow_rasterizer : shadow_rasterizer);

			for (auto chunk : visible_chunks) {
				auto chunk_position = get_chunk_position(chunk);
				auto lod_index = (u32)get_chunk_lod_t(chunk_position);

				if (chunk->trees_instances_buffer.count) {
					auto relative_position = (v3f)(chunk_position - current_world_origin) * CHUNKW;

					auto &lod = tree_model.get_lod(lod_index);
					if (!lod.cast_shadow)
						continue;

					entity_cbuffer.update({
						.relative_position = relative_position,
						.vertex_offset = lod.start_vertex,
						.lod_t = chunk->lod_t,
					});

					immediate_context->VSSetShaderResources(INSTANCE_BUFFER_SLOT, 1, &chunk->trees_instances_buffer.srv);

					immediate_context->DrawIndexedInstanced(lod.index_count, chunk->trees_instances_buffer.count, lod.start_index, lod.start_vertex, 0);
					draw_call_count++;

					//if (chunk->lod_t < 1) {
					//	entity_cbuffer.update({
					//		.relative_position = relative_position,
					//		.actual_position = (v3f)(chunk_position*CHUNKW),
					//		.lod_t = -chunk->lod_t,
					//	});
					//
					//	auto &model = tree_model.get_lod(chunk->previous_lod);
					//	immediate_context->VSSetShaderResources(VERTEX_BUFFER_SLOT, 1, &model.vb);
					//	immediate_context->VSSetShaderResources(INSTANCE_BUFFER_SLOT, 1, &chunk->trees_instances_buffer.srv);
					//	immediate_context->PSSetShaderResources(ALBEDO_TEXTURE_SLOT, 1, &model.albedo);
					//	immediate_context->RSSetState(model.no_cull ? no_cull_rasterizer : 0);
					//	immediate_context->IASetIndexBuffer(model.ib, DXGI_FORMAT_R32_UINT, 0);
					//	immediate_context->DrawIndexedInstanced(model.index_count, chunk->trees_instances_buffer.count, 0, 0, 0);
					//}
				}
			}
		}
		//
		// ENTITIES SHADOW
		//
		{
			scoped_gpu_timer("entities shadow");

			scoped_allocator(temporary_allocator);
			HashMap<LodModel *, List<Entity *>> entities_to_draw;

			for (auto &mesh : components<MeshComponent>) {
				entities_to_draw.get_or_insert(mesh.model).add(mesh.entity);
			}

			immediate_context->VSSetShader(shaders.model_shadow_vs, 0, 0);
			immediate_context->PSSetShader(shaders.model_shadow_ps, 0, 0);
			immediate_context->PSSetSamplers(DEFAULT_SAMPLER_SLOT, 1, &default_sampler_wrap);
			// immediate_context->OMSetBlendState(alpha_to_coverage_blend, {}, -1);

			for_each(entities_to_draw, [&](LodModel *model, List<Entity *> entities){
				immediate_context->VSSetShaderResources(VERTEX_BUFFER_SLOT,  1, &model->vertex_buffer);
				immediate_context->PSSetShaderResources(ALBEDO_TEXTURE_SLOT, 1, model->albedo ? &model->albedo : &default_albedo);
				immediate_context->PSSetShaderResources(NORMAL_TEXTURE_SLOT, 1, model->normal ? &model->normal : &default_normal);
				immediate_context->PSSetShaderResources(AO_TEXTURE_SLOT,     1, model->ao     ? &model->ao     : &default_ao    );
				immediate_context->RSSetState(model->no_cull ? no_cull_shadow_rasterizer : shadow_rasterizer);
				immediate_context->IASetIndexBuffer(model->index_buffer, DXGI_FORMAT_R32_UINT, 0);

				for (auto entity : entities) {
					entity_cbuffer.update({
						.model = to_m4(transpose(std::bit_cast<m3>(entity->rotation.getMatrix()))),
						.relative_position = entity->position.local + (v3f)(entity->position.chunk - current_world_origin)*CHUNKW,
						.vertex_offset = model->lods[0].start_vertex,
					});

					draw_indexed(model->lods[0].index_count, model->lods[0].start_index, model->lods[0].start_vertex);
				}
			});
		}


		immediate_context->OMSetRenderTargets(0, 0, 0);
		immediate_context->PSSetShaderResources(SHADOW_TEXTURE_SLOT, 1, &shadow_srv);


		frame_cbuffer.update({
			.mvp = camera_matrix,
			.rotproj = rotproj,
			.light_vp_matrix = light_vp_matrix,
			.campos = camera_position_relative_to_origin,
			.time = (f32)get_time(the_timer),
			.ldir = light_dir,
			.frame = (f32)frame_number,
			.draw_mode = draw_mode,
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
		scoped_gpu_timer("sky render");

		immediate_context->VSSetShader(shaders.sky_vs, 0, 0);
		immediate_context->PSSetShader(shaders.sky_ps, 0, 0);
		ID3D11ShaderResourceView *null = 0;
		immediate_context->PSSetShaderResources(SKY_TEXTURE_SLOT, 1, &null);
		immediate_context->RSSetState(0);
		immediate_context->OMSetRenderTargets(1, &sky_rt, 0);
		immediate_context->OMSetBlendState(0, {}, -1);
		draw_vertices(36);

		immediate_context->OMSetRenderTargets(1, &back_buffer, 0);
		immediate_context->ClearDepthStencilView(depth_stencil, D3D11_CLEAR_DEPTH, 1, 0);

		immediate_context->PSSetShaderResources(SKY_TEXTURE_SLOT, 1, &sky_srv);
	}

	//
	// SKY BLIT
	//
	{
		scoped_gpu_timer("sky blit");
		immediate_context->VSSetShader(shaders.blit_vs, 0, 0);
		immediate_context->PSSetShader(shaders.blit_ps, 0, 0);
		immediate_context->RSSetState(0);
		immediate_context->OMSetRenderTargets(1, &back_buffer, 0);
		immediate_context->OMSetBlendState(0, {}, -1);
		draw_vertices(6);
	}

	//
	// FARLANDS
	//

	{
		scoped_gpu_timer("draw far lands");
		immediate_context->OMSetRenderTargets(1, &back_buffer, depth_stencil);
		immediate_context->OMSetBlendState(0, {}, -1);
		immediate_context->VSSetShader(shaders.chunk_sdf_vs, 0, 0);
		immediate_context->PSSetShader(shaders.chunk_sdf_solid_ps, 0, 0);

		immediate_context->PSSetShaderResources(ALBEDO_TEXTURE_SLOT, 1, &ground_albedo);
		immediate_context->PSSetShaderResources(NORMAL_TEXTURE_SLOT, 1, &ground_normal);
		immediate_context->RSSetState(0);

		if (farlands_vb.vertex_count) {
			entity_cbuffer.update({
				.random = random_v4f({}),
				.vertex_offset = 0,
			});

			immediate_context->VSSetShaderResources(VERTEX_BUFFER_SLOT, 1, &farlands_vb.view);
			draw_vertices(farlands_vb.vertex_count);
		}
		immediate_context->ClearDepthStencilView(depth_stencil, D3D11_CLEAR_DEPTH, 1, 0);
	}

	//
	// CHUNKS SDF MESH
	//

	visible_chunks.clear();
	{
		scoped_gpu_timer("draw chunks sdf mesh");
		immediate_context->OMSetRenderTargets(1, &back_buffer, depth_stencil);
		immediate_context->OMSetBlendState(0, {}, -1);
		immediate_context->VSSetShader(shaders.chunk_sdf_vs, 0, 0);
		immediate_context->PSSetShader(shaders.chunk_sdf_solid_ps, 0, 0);

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

			auto surface_lod_t = get_chunk_lod_t(chunk_position);
			auto lod_index = debug_selected_lod == -1 ? (u32)surface_lod_t : debug_selected_lod;

			if (chunk->lod_previous_frame != lod_index) {
				if (lod_index == chunk->previous_lod) {
					// NOTE: If we are transitioning from lod0 to lod1 and go back to lod0, make it smooth.
					// One thing that is not perfect is that lod mask flips noticeably.
					chunk->lod_t = max(0, 1 - chunk->lod_t);
				} else {
					chunk->lod_t = 0;
				}
				chunk->previous_lod = chunk->lod_previous_frame;
				chunk->lod_previous_frame = lod_index;
			}

			// NOTE: If lod_t is zero, the shader doesn't not which way to blend.
			// So add frame time after setting it to 0.
			chunk->lod_t += frame_time;

			if (chunk_in_view(camera_frustum, chunk_position)) {
				visible_chunks.add(chunk);

				if (chunk->sdf_vb_vertex_count[lod_index]) {
					auto relative_position = (v3f)(chunk_position - current_world_origin) * CHUNKW;
					entity_cbuffer.update({
						.random = random_v4f(chunk_position),
						.relative_position = relative_position,
						.was_remeshed = (f32)(chunk->frames_since_remesh < 5),
						.vertex_offset = chunk->sdf_vb_vertex_offset[lod_index],
						.lod_t = surface_lod_t - lod_index,
					});

					immediate_context->VSSetShaderResources(VERTEX_BUFFER_SLOT, 1, &chunk->sdf_vb);
					draw_vertices(chunk->sdf_vb_vertex_count[lod_index]);
				}
			}
		}}}
	}

	//
	// CHUNKS BLOCKS MESH
	//
	{
		scoped_gpu_timer("draw chunks blocks mesh");
		immediate_context->OMSetRenderTargets(1, &back_buffer, depth_stencil);
		immediate_context->OMSetBlendState(0, {}, -1);
		immediate_context->VSSetShader(shaders.chunk_block_vs, 0, 0);
		immediate_context->PSSetShader(shaders.chunk_block_ps, 0, 0);

		immediate_context->PSSetShaderResources(ALBEDO_TEXTURE_SLOT, 1, &planks_albedo);
		immediate_context->PSSetShaderResources(NORMAL_TEXTURE_SLOT, 1, &planks_normal);
		immediate_context->RSSetState(0);

		for (auto chunk : visible_chunks) {
			auto chunk_position = get_chunk_position(chunk);

			auto &vertex_buffer = chunk->blocks_vb;
			if (vertex_buffer.view) {
				auto relative_position = (v3f)(chunk_position - current_world_origin) * CHUNKW;
				entity_cbuffer.update({
					.random = random_v4f(chunk_position),
					.relative_position = relative_position,
				});

				immediate_context->VSSetShaderResources(VERTEX_BUFFER_SLOT, 1, &vertex_buffer.view);
				draw_vertices(vertex_buffer.vertex_count);
			}
		}
	}

	//
	// CHUNKS GRASS
	//
	if (draw_grass) {
		scoped_gpu_timer("draw chunks grass");
		immediate_context->RSSetState(no_cull_rasterizer);
		immediate_context->VSSetShader(shaders.grass_vs, 0, 0);
		immediate_context->PSSetShader(shaders.grass_ps, 0, 0);

		immediate_context->PSSetShaderResources(ALBEDO_TEXTURE_SLOT, 1, &grass_albedo);
		immediate_context->PSSetShaderResources(NORMAL_TEXTURE_SLOT, 1, &grass_normal);
		immediate_context->OMSetBlendState(0, {}, -1);

		for (auto &chunk : visible_chunks) {
			auto chunk_position = get_chunk_position(chunk);
			auto lod_index = (u32)get_chunk_lod_t(chunk_position);
			if (lod_index <= 1) {
				if (chunk->grass_vb.view) {
					auto relative_position = (v3f)(chunk_position - current_world_origin) * CHUNKW;
					entity_cbuffer.update({
						.random = random_v4f(chunk_position + 0x12345678),
						.relative_position = relative_position,
					});

					immediate_context->VSSetShaderResources(VERTEX_BUFFER_SLOT, 1, &chunk->grass_vb.view);
					draw_vertices(chunk->grass_vb.vertex_count);
				}
			}
		}
	}

	// TREE
	if (draw_trees) {
		scoped_gpu_timer("trees surface");
		immediate_context->VSSetShader(shaders.tree_vs, 0, 0);
		immediate_context->PSSetShader(shaders.tree_ps, 0, 0);
		immediate_context->PSSetSamplers(DEFAULT_SAMPLER_SLOT, 1, &default_sampler_wrap);
		// immediate_context->OMSetBlendState(alpha_to_coverage_blend, {}, -1);

		immediate_context->VSSetShaderResources(VERTEX_BUFFER_SLOT, 1, &tree_model.vertex_buffer);
		immediate_context->PSSetShaderResources(ALBEDO_TEXTURE_SLOT, 1, &tree_model.albedo);
		immediate_context->PSSetShaderResources(NORMAL_TEXTURE_SLOT, 1, &tree_model.normal);
		immediate_context->PSSetShaderResources(AO_TEXTURE_SLOT,     1, &tree_model.ao);
		immediate_context->RSSetState(tree_model.no_cull ? no_cull_rasterizer : 0);
		immediate_context->IASetIndexBuffer(tree_model.index_buffer, DXGI_FORMAT_R32_UINT, 0);

		for (auto &chunk : visible_chunks) {
			auto chunk_position = get_chunk_position(chunk);
			auto lod_index = (u32)get_chunk_lod_t(chunk_position);

			if (chunk->trees_instances_buffer.count) {
				auto relative_position = (v3f)(chunk_position - current_world_origin) * CHUNKW;

				auto &lod = tree_model.get_lod(lod_index);

				entity_cbuffer.update({
					.random = random_v4f(chunk_position + 0x87654321),
					.relative_position = relative_position,
					.vertex_offset = lod.start_vertex,
					.lod_t = chunk->lod_t,
				});

				immediate_context->VSSetShaderResources(INSTANCE_BUFFER_SLOT, 1, &chunk->trees_instances_buffer.srv);
				immediate_context->DrawIndexedInstanced(lod.index_count, chunk->trees_instances_buffer.count, lod.start_index, lod.start_vertex, 0);
				draw_call_count++;

				if (chunk->lod_t < 1) {
					auto &lod = tree_model.get_lod(chunk->previous_lod);

					entity_cbuffer.update({
						.random = random_v4f(chunk_position + 0x87654321),
						.relative_position = relative_position,
						.vertex_offset = lod.start_vertex,
						.lod_t = -chunk->lod_t,
					});

					immediate_context->DrawIndexedInstanced(lod.index_count, chunk->trees_instances_buffer.count, lod.start_index, lod.start_vertex, 0);
					draw_call_count++;
				}
			}
		}
	}

	immediate_context->RSSetState(0);
	immediate_context->PSSetSamplers(DEFAULT_SAMPLER_SLOT, 1, &default_sampler_wrap);

	if (key_down('R'))
		wireframe_rasterizer_enabled = !wireframe_rasterizer_enabled;

	immediate_context->RSSetState(wireframe_rasterizer);
	immediate_context->VSSetShader(shaders.chunk_sdf_vs, 0, 0);
	immediate_context->PSSetShader(shaders.chunk_sdf_wire_ps, 0, 0);
	immediate_context->OMSetBlendState(0, {}, -1);

	if (wireframe_rasterizer_enabled) {
		for (auto &chunk : visible_chunks) {
			auto chunk_position = get_chunk_position(chunk);

			f32 lod_t = get_chunk_lod_t(chunk_position);
			auto lod_index = debug_selected_lod == -1 ? (u32)lod_t : debug_selected_lod;

			if (chunk->sdf_vb_vertex_count[lod_index]) {
				auto relative_position = (v3f)(chunk_position - current_world_origin) * CHUNKW;
				entity_cbuffer.update({
					.relative_position = relative_position,
					.vertex_offset = chunk->sdf_vb_vertex_offset[lod_index],
					.lod_t = lod_t - lod_index,
				});

				immediate_context->VSSetShaderResources(VERTEX_BUFFER_SLOT, 1, &chunk->sdf_vb);
				draw_vertices(chunk->sdf_vb_vertex_count[lod_index]);
			}
		}
	}

	//
	// ENTITIES
	//
	{
		scoped_gpu_timer("entities");

		scoped_allocator(temporary_allocator);
		HashMap<LodModel *, List<Entity *>> entities_to_draw;

		for (auto &mesh : components<MeshComponent>) {
			entities_to_draw.get_or_insert(mesh.model).add(mesh.entity);
		}

		immediate_context->VSSetShader(shaders.model_vs, 0, 0);
		immediate_context->PSSetShader(shaders.model_ps, 0, 0);
		immediate_context->PSSetSamplers(DEFAULT_SAMPLER_SLOT, 1, &default_sampler_wrap);
		// immediate_context->OMSetBlendState(alpha_to_coverage_blend, {}, -1);

		for_each(entities_to_draw, [&](LodModel *model, List<Entity *> entities){
			immediate_context->VSSetShaderResources(VERTEX_BUFFER_SLOT,  1, &model->vertex_buffer);
			immediate_context->PSSetShaderResources(ALBEDO_TEXTURE_SLOT, 1, model->albedo ? &model->albedo : &default_albedo);
			immediate_context->PSSetShaderResources(NORMAL_TEXTURE_SLOT, 1, model->normal ? &model->normal : &default_normal);
			immediate_context->PSSetShaderResources(AO_TEXTURE_SLOT,     1, model->ao     ? &model->ao     : &default_ao    );
			immediate_context->RSSetState(model->no_cull ? no_cull_rasterizer : 0);
			immediate_context->IASetIndexBuffer(model->index_buffer, DXGI_FORMAT_R32_UINT, 0);

			for (auto entity : entities) {
				entity_cbuffer.update({
					.model = to_m4(transpose(std::bit_cast<m3>(entity->rotation.getMatrix()))),
					.relative_position = entity->position.local + (v3f)(entity->position.chunk-current_world_origin)*CHUNKW,
					.vertex_offset = model->lods[0].start_vertex,
				});

				draw_indexed(model->lods[0].index_count, model->lods[0].start_index, model->lods[0].start_vertex);
			}
		});
	}

	//
	// DRAW DEBUG PHYSICS
	//
	{
		auto &debug = physics_world->getDebugRenderer();

		auto &triangles = debug.getTriangles();
		if (triangles.size()) {
			struct Vertex {
				v3f position;
				u32 color;
			};

			auto vertices = Span((Vertex *)&triangles[0], triangles.size()*3);

			static ID3D11ShaderResourceView *debug_buffer = 0;
			if (debug_buffer) {
				debug_buffer->Release();
				debug_buffer = 0;
			}
			sizeof(triangles[0]);
			debug_buffer = create_structured_buffer(vertices);

			immediate_context->RSSetState(wireframe_rasterizer);
			immediate_context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
			immediate_context->VSSetShader(shaders.collider_vs, 0, 0);
			immediate_context->PSSetShader(shaders.collider_ps, 0, 0);
			immediate_context->VSSetShaderResources(VERTEX_BUFFER_SLOT, 1, &debug_buffer);
			draw_vertices(vertices.count);
		}

		auto &lines = debug.getLines();
		if (lines.size()) {
			static ID3D11ShaderResourceView *debug_buffer = 0;
			if (debug_buffer) {
				debug_buffer->Release();
				debug_buffer = 0;
			}
			debug_buffer = create_structured_buffer(Span(&lines[0], lines.size()));

			immediate_context->RSSetState(wireframe_rasterizer);
			immediate_context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_LINELIST);
			immediate_context->VSSetShader(shaders.collider_vs, 0, 0);
			immediate_context->PSSetShader(shaders.collider_ps, 0, 0);
			immediate_context->VSSetShaderResources(VERTEX_BUFFER_SLOT, 1, &debug_buffer);
			draw_vertices(lines.size()*2);
		}
	}


	immediate_context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_LINELIST);
	immediate_context->RSSetState(0);
	immediate_context->OMSetRenderTargets(1, &back_buffer, 0);
	immediate_context->OMSetBlendState(0, {}, -1);
	immediate_context->VSSetShader(shaders.crosshair_vs, 0, 0);
	immediate_context->PSSetShader(shaders.crosshair_ps, 0, 0);
	draw_vertices(4);


	{
		timed_block(profiler, profile_frame, "ui");

		u32 const font_size = 16;

		struct Label {
			v2f position;
			v2f size;
			Span<utf8> text;
		};
		StaticList<Label, 1024*16> labels;

		//
		// PROFILE
		//
		if (show_profile) {
			immediate_context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
			immediate_context->OMSetBlendState(alpha_blend, v4f{}.s, -1);
			immediate_context->VSSetShader(shaders.color_vs, 0, 0);
			immediate_context->PSSetShader(shaders.color_ps, 0, 0);
			auto rect = [&](v2f position, v2f size, v4f color) {
				D3D11_VIEWPORT viewport {
					.TopLeftX = clamp<f32>(position.x, 0, window_client_size.x),
					.TopLeftY = position.y,
					.Width = clamp<f32>(size.x, 0, window_client_size.x - position.x),
					.Height = size.y,
					.MinDepth = 0,
					.MaxDepth = 1,
				};
				immediate_context->RSSetViewports(1, &viewport);
				entity_cbuffer.update({
					.random = color,
				});
				draw_vertices(6);
			};

			auto get_groups_from_sorted = [&]<class T, umm capacity, class Fn>(StaticList<Span<T>, capacity> *destination, Span<T> source, Fn &&property) {
				if (!source.count)
					return;

				umm first_index = 0;
				umm current_index = 0;

				auto current_property = property(source.data[0]);

				++current_index;

				while (current_index < source.count) {
					auto new_property = property(source.data[current_index]);
					if (new_property != current_property) {
						destination->add({source.data + first_index, source.data + current_index});

						current_property = new_property;
						first_index = current_index;
					}

					++current_index;
				}
				destination->add({source.data + first_index, source.data + current_index});
			};

			StaticList<Span<Profiler::TimeSpan>, 8> thread_spans;
			get_groups_from_sorted(&thread_spans, previous_frame_profile, [&](Profiler::TimeSpan a) { return a.thread_id; });

			f32 const thread_stride = font_size*8;
			u32 thread_index = 0;
			for (auto &thread_span : thread_spans) {
				defer { ++thread_index; };

				StaticList<u64, 256> end_stack;
				u32 deepness = 0;

				for (auto span : thread_span) {
					defer { end_stack.add(span.end); };
					if (end_stack.count) {
						while (end_stack.count && span.end >= end_stack.back()) {
							end_stack.pop_back();
							--deepness;
						}
						++deepness;
					}

					f64 const x_scale = 0.5 * (f64)window_client_size.x / performance_frequency / target_frame_time;
					f32 x = span.begin * x_scale;
					if (x >= window_client_size.x)
						continue;

					f32 w = max(1, (f32)(span.end - span.begin) * x_scale);
					f32 h = font_size;
					f32 y = 350.f+thread_index*thread_stride + deepness*h;

					labels.add({{x,y}, {w,h}, span.name});
					rect(
						{x,y},
						{w,h},
						V4f(random_v3f(get_hash(span.name)), .5f)
					);
				}
			}
			struct NamedGpuTime {
				GpuTime time;
				char const *name;
			};
			StaticList<NamedGpuTime, 256> gpu_times;

			u64 start_tick = -1;
			for (auto query : gpu_time_queries) {
				auto time = get_gpu_time(query);
				start_tick = min(start_tick, time.start);
				gpu_times.add({time, query.name});
			}

			for (auto _time : gpu_times) {
				auto time = _time.time;
				auto name = _time.name;
				f64 const x_scale = 0.5 * (f64)window_client_size.x / time.frequency / target_frame_time;

				f32 x = (time.start - start_tick) * x_scale;
				if (x >= window_client_size.x)
					continue;

				f32 w = max(1, (f32)(time.end - time.start) * x_scale);
				f32 h = font_size;
				f32 y = 350.f+thread_index*thread_stride;


				auto _name = as_utf8(as_span(name));
				labels.add({{x,y}, {w,h}, _name});
				rect(
					{x,y},
					{w,h},
					V4f(random_v3f(get_hash(_name)), .5f)
				);
			}
		}


		//
		// DEBUG TEXT
		//

		D3D11_VIEWPORT viewport {
			.TopLeftX = 0,
			.TopLeftY = 0,
			.Width = (f32)window_client_size.x,
			.Height = (f32)window_client_size.y,
			.MinDepth = 0,
			.MaxDepth = 1,
		};
		immediate_context->RSSetViewports(1, &viewport);

		StringBuilder builder;
		builder.allocator = temporary_allocator;

		append_format(builder, u8R"(frame time: {} ms
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
draw_call_count: {}
profile:
)"s,
frame_time * 1000,
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
remesh_count,
draw_call_count
);
		// HashMap<Span<utf8>, u64> time_spans;
		// time_spans.allocator = temporary_allocator;
		//
		// for (auto &span : profiler.recorded_time_spans) {
		// 	time_spans.get_or_insert(span.name) += span.end - span.begin;
		// }
		// for_each(time_spans, [&] (auto name, auto duration) {
		// 	append_format(builder, "  {} {}us\n", name, duration * 1'000'000 / performance_frequency);
		// });

		append_format(builder, "chunk info:\n{}", chunk_info_string);
		append_format(builder, "allocation info:\n{}", allocation_info_string);

		auto string = to_string(builder);
		defer { free(string); };

		draw_text((Span<utf8>)string, font_size, {});

		immediate_context->RSSetState(scissor_rasterizer);
		for (auto label : labels) {
			D3D11_RECT scissor {
				.left = (LONG)label.position.x,
				.top = (LONG)label.position.y,
				.right = (LONG)(label.position.x + label.size.x),
				.bottom = (LONG)(label.position.y + label.size.y),
			};
			immediate_context->RSSetScissorRects(1, &scissor);
			draw_text(label.text, font_size, label.position);
		}
	}
	immediate_context->RSSetState(0);
#if 0
	for (auto &chunk : flatten(chunks)) {
		auto chunk_position = get_chunk_position(&chunk);
		auto &vertex_buffer = chunk.sdf_vertex_buffers[debug_selected_lod == -1 ? (u32)get_chunk_lod_t(chunk_position) : debug_selected_lod];
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

auto float_error(auto value) {
	return nextafter(value, value * 2) - value;
}

s32 tl_main(Span<Span<utf8>> arguments) {
	printf("%f\n", float_error((f32)CHUNKW) / float_error((f64)CHUNKW*max_value<s32>+CHUNKW));
	init_printer();

	// {
	// 	m4 a = *(m4 *)argv[0];
	// 	m4 b = *(m4 *)argv[1];
	// 	m4 c = a * b;
	// 	*(m4 *)argv[1] = c;
	// }

	HINSTANCE hInstance = GetModuleHandleW(0);

	//init_tracking_allocator();
	//current_allocator = tracking_allocator;
	//tracking_allocator_fallback = os_allocator;

	executable_path = get_executable_path();
	executable_directory = parse_path(executable_path).directory;

	_chunks = (decltype(_chunks))VirtualAlloc(0, sizeof(chunks[0][0][0]) * pow3(DRAWD*2+1), MEM_COMMIT|MEM_RESERVE, PAGE_READWRITE);
	for (auto &chunk : flatten(chunks)) {
		construct(chunk.physics_vertices);
		construct(chunk.physics_indices);
		construct(chunk.trees);
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
	construct(previous_frame_profile);
	construct(gpu_time_queries);
#define e(name) construct(components<name>);
ENUMERATE_COMPONENTS
#undef e

	thread_count = get_cpu_info().logical_processor_count;
	//thread_count = 1;
	init_thread_pool(thread_pool, thread_count-1); // main thread is also a worker

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
	if (arguments.count != 1) {
		opponent_socket = net::create_socket(net::Connection_tcp);

		if (!net::connect(opponent_socket, inet_addr((char *)arguments[1].data), 27015) != 0) {
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
#if D3D11_DEBUG
	dhr(D3D11CreateDeviceAndSwapChain(0, D3D_DRIVER_TYPE_HARDWARE, 0, D3D11_CREATE_DEVICE_DEBUG, &feature, 1, D3D11_SDK_VERSION, &sd, &swap_chain, &device, 0, &immediate_context));
	dhr(device->QueryInterface(&debug_info_queue));
#else
	dhr(D3D11CreateDeviceAndSwapChain(0, D3D_DRIVER_TYPE_HARDWARE, 0, 0, &feature, 1, D3D11_SDK_VERSION, &sd, &swap_chain, &device, 0, &immediate_context));
#endif

	resize();
	void compile_shaders();
	compile_shaders();


	frame_cbuffer.init();
	entity_cbuffer.init();

	{
		u32 albedo[] {
			0xffffffff
		};
		default_albedo = make_texture(albedo);
	}
	{
		u32 normal[] { 0xffff8080 };
		default_normal = make_texture(normal);
	}
	{
		u32 ao[] { 0xffffffff };
		default_ao = make_texture(ao);
	}

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
			.FillMode = D3D11_FILL_SOLID,
			.CullMode = D3D11_CULL_BACK,
			.ScissorEnable = true,
		};
		dhr(device->CreateRasterizerState(&desc, &scissor_rasterizer));
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
			.Filter = D3D11_FILTER_MIN_MAG_MIP_POINT,
			.AddressU = D3D11_TEXTURE_ADDRESS_MIRROR,
			.AddressV = D3D11_TEXTURE_ADDRESS_MIRROR,
			.AddressW = D3D11_TEXTURE_ADDRESS_MIRROR,
			.MaxAnisotropy = 16,
			.MinLOD = 0,
			.MaxLOD = max_value<f32>,
		};

		ID3D11SamplerState *sampler;
		dhr(device->CreateSamplerState(&desc, &sampler));

		immediate_context->PSSetSamplers(NEAREST_MIRROR_SAMPLER_SLOT, 1, &sampler);
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
		u8 pixels[64][64];
		u8 i = 0;
		for (auto &p : flatten(pixels)) {
			p = i++;
		}

		std::shuffle(flatten(pixels).begin(), flatten(pixels).end(), std::random_device{});

		lod_mask = make_texture(pixels, false);
		immediate_context->PSSetShaderResources(LOD_MASK_TEXTURE_SLOT, 1, &lod_mask);
	}

	auto load_model = [&](char const *name) {
		scoped_allocator(temporary_allocator);
		auto scene = parse_glb_from_memory(read_entire_file(format("{}/{}", executable_directory, name)));

		List<Scene3D::Node *> lod_nodes;
		List<BasicVertex> vertices;
		List<u32> indices;

		defer {
			free(lod_nodes);
			free(vertices);
			free(indices);
		};

		LodModel result = {};

		auto add_lod = [&](Scene3D::Node *node, LodModel::Lod lod = {}) {
			auto mesh = node->mesh;

			u32 index_count = mesh->indices.count;
			u32 start_index  = indices.count;
			u32 start_vertex = vertices.count;

			lod.index_count = index_count;
			lod.start_index  = start_index;
			lod.start_vertex = start_vertex;
			result.lods.add(lod);

			for (auto v : mesh->vertices) {
				vertices.add({
					.position = v.position,
					.normal   = v.normal,
					.uv       = v.uv,
					.tangent  = v.tangent,
				});
			}
			indices.add(mesh->indices);
		};

		for (auto &node : scene.nodes) {
			if (starts_with(node.name, u8"lod"s)) {
				lod_nodes.add(&node);
			}
		}

		for (auto node : lod_nodes) {
			add_lod(node, {
				.end_distance = (u32)parse_u64(node->name.subspan(3, 1)).value(),
				.cast_shadow = node->name[4] == 'y',
			});
		}

		std::sort(result.lods.begin(), result.lods.end(), [&](LodModel::Lod a, LodModel::Lod b) {
			return a.end_distance < b.end_distance;
		});

		result.vertex_buffer = create_structured_buffer(vertices);
		result.index_buffer  = create_index_buffer(indices);

		result.no_cull = true;

		return result;
	};

	tree_model = load_model("spruce.glb");
	grenade_model = load_model("grenade.glb");

	struct ReloadableTexture {
		char const *path;
		FileTime last_write_time;
		ID3D11ShaderResourceView **srv;
	};

	StaticList<ReloadableTexture, 16> textures;

	textures.add({.path = format("{}\\ground_albedo.png\0"s, executable_directory).data, .srv = &ground_albedo});
	textures.add({.path = format("{}\\ground_normal.png\0"s, executable_directory).data, .srv = &ground_normal});
	textures.add({.path = format("{}\\grass.png\0"s, executable_directory).data, .srv = &grass_albedo});
	textures.add({.path = format("{}\\planks.png\0"s, executable_directory).data, .srv = &planks_albedo});
	textures.add({.path = format("{}\\planks_normal.png\0"s, executable_directory).data, .srv = &planks_normal});
	textures.add({.path = format("{}\\spruce_albedo.png\0"s, executable_directory).data, .srv = &tree_model.albedo});
	textures.add({.path = format("{}\\spruce_normal.png\0"s, executable_directory).data, .srv = &tree_model.normal});
	textures.add({.path = format("{}\\spruce_ao.png\0"s, executable_directory).data,     .srv = &tree_model.ao});

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

	p::PhysicsWorld::WorldSettings settings;
	settings.defaultBounciness = 0;
	physics_world = physics.createPhysicsWorld(settings);
	physics_world->setIsDebugRenderingEnabled(true);

	camera = create_entity();
	camera->position = {0,-3,0};
	auto &camera_physics = camera->add_component<PhysicsComponent>();
	camera_physics.collider = camera_physics.body->addCollider(physics.createCapsuleShape(.2, 1), {});
	camera_physics.body->setAngularLockAxisFactor({1,1,1});

	start();

	target_frame_time = 1 / 75.0f;

	lock_cursor();

	make_os_timing_precise();
	auto frame_time_counter = get_performance_counter();
	auto frame_timer = create_precise_timer();
	auto stat_reset_timer = get_performance_counter();
	the_timer = create_precise_timer();

    while (1) {
		for (auto &t : textures) {
			auto status = detect_change(t.path, &t.last_write_time);
			if (status.failed) {
				print("detect status failed {}\n", t.path);
			}
			if (status.changed) {
				int w,h;
				auto pixels = stbi_load(t.path, &w, &h, 0, 4);
				defer { stbi_image_free(pixels); };
				if (*t.srv)
					(*t.srv)->Release();
				*t.srv = make_texture(pixels, w, h);
			}
		}

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


		if (key_down('G')) {
			show_profile = !show_profile;
		}

		profile_frame = key_held('T');
		if (profile_frame) {
			show_profile = true;
			profiler.reset();
		}
		defer {
			if (profile_frame) {
				previous_frame_profile.set(profiler.recorded_time_spans);
				for (auto &span : previous_frame_profile) {
					assert(span.begin >= profiler.start_time);
					assert(span.end   >= profiler.start_time);
					span.begin -= profiler.start_time;
					span.end   -= profiler.start_time;
				}

				std::sort(previous_frame_profile.begin(), previous_frame_profile.end(), [&](Profiler::TimeSpan a, Profiler::TimeSpan b) {
					if (a.thread_id != b.thread_id)
						return a.thread_id < b.thread_id;

					if (a.begin != b.begin)
						return a.begin < b.begin;

					return a.end < b.end;
				});

				write_entire_file("frame.tmd"s, (Span<u8>)profiler.output_for_timed());
			}
			profile_frame = false;
		};

		timed_block(profiler, profile_frame, "frame");

		update();
		swap_chain->Present(0, 0);

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
