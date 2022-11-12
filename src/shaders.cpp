#include "shaders.h"

Shaders shaders;

void compile_shaders() {
	shaders.chunk_sdf_vs = create_vs(HLSL_CBUFFER HLSL_COMMON R"(
struct ChunkVertex {
	float3 position;
	uint normal;
	float3 parent_position;
	uint parent_normal;
};

StructuredBuffer<ChunkVertex> s_vertex_buffer : VERTEX_BUFFER_SLOT;

void main(
	in uint vertex_id : SV_VertexID,

	out float3 normal : NORMAL,
	out float3 color : COLOR,
	out float3 wpos : WPOS,
	out float3 view : VIEW,
	out float4 shadow_map_uv : SHADOW_MAP_UV,
	out float4 position : SV_Position
) {
	ChunkVertex vertex = s_vertex_buffer[vertex_id+c_vertex_offset];

	float3 pos = vertex.position + c_relative_position;
	normal = decode_normal(vertex.normal);
#if 0
	pos = lerp(
		pos,
		vertex.parent_position + c_relative_position,
		c_lod_t
	);
	normal = lerp(
		normal,
		decode_normal(vertex.parent_normal),
		c_lod_t
	);
#endif

	wpos = pos;
	view = c_campos - pos;

	if (c_was_remeshed == 0) {
		color = 1;
	} else {
		color = float3(1, 0, 0);
	}

	position = mul(c_mvp, float4(pos, 1.0f));

	shadow_map_uv = mul(light_vp_matrix, float4(pos, 1));
}
)"s);

	shaders.chunk_sdf_solid_ps = create_ps(HLSL_CBUFFER HLSL_COMMON HLSL_LIGHTING HLSL_TRIPLANAR R"(
void main(
	in float3 normal : NORMAL,
	in float3 color_ : COLOR,
	in float3 wpos : WPOS,
	in float3 view : VIEW,
	in float4 shadow_map_uv : SHADOW_MAP_UV,
	in float4 pixel_position : SV_Position,

	out float4 pixel_color : SV_Target
) {
	normal = normalize(normal);

	float3 N = float4(triplanar_normal(normal_texture, default_sampler, wpos/32, normal), 1);

	float trip = triplanar(albedo_texture, default_sampler, wpos/32, normal).x;

	float3 grass = float3(.29,.69,.14);
	float3 rock  = float3(.2,.2,.1);

	float gr = smoothstep(0.3, 0.7, normal.y);
	gr = lerp(gr, trip, (0.5f - abs(0.5f - gr)) * 2);

	float3 albedo = lerp(rock, grass, gr) * trip;// * color_;

	float metalness = 0;
	float roughness = 1;

	pixel_color = surface(N, c_ldir, view, shadow_map_uv, albedo, 1, roughness, pixel_position);
}
)"s);


	shaders.chunk_sdf_wire_ps = create_ps(R"(
void main(in float3 normal : NORMAL, out float4 pixel_color : SV_Target) {
	pixel_color.rgb = 0;
	pixel_color.a = 1;
}
)"s);

	shaders.cursor_vs = create_vs(HLSL_CBUFFER R"(
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

	position = mul(c_mvp, mul(c_model, float4(positions[vertex_id], 1)) + float4(c_relative_position, 0));
	color = positions[vertex_id];
}
)"s);
	shaders.cursor_ps = create_ps(R"(
void main(in float3 color: COLOR, out float4 pixel_color : SV_Target) {
	pixel_color = float4(color, 1);
}
)"s);

	shaders.sky_vs = create_vs(HLSL_CBUFFER R"(
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
	shaders.sky_ps = create_ps(HLSL_CBUFFER HLSL_COMMON R"(
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

	shaders.blit_vs = create_vs(R"(
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
	shaders.blit_ps = create_ps(HLSL_CBUFFER R"(
void main(in float2 uv : UV, out float4 pixel_color : SV_Target) {
	pixel_color = sky_texture.Sample(default_sampler, uv);
}
)"s);

	shaders.shadow_vs = create_vs(HLSL_CBUFFER HLSL_COMMON R"(
struct ChunkVertex {
	float3 position;
	uint normal;
	float3 parent_position;
	uint parent_normal;
};

StructuredBuffer<ChunkVertex> s_vertex_buffer : VERTEX_BUFFER_SLOT;

void main(
	in uint vertex_id : SV_VertexID,
	out float4 position : SV_Position
) {
	ChunkVertex vertex = s_vertex_buffer[vertex_id+c_vertex_offset];

	float3 pos = vertex.position + c_relative_position;
#if 0
	pos = lerp(
		pos,
		vertex.parent_position + c_relative_position,
		c_lod_t
	);
#endif

	position = mul(c_mvp, float4(pos, 1.0f));
}
)"s);

	shaders.shadow_ps = create_ps(HLSL_CBUFFER HLSL_COMMON R"(
void main() {
}
)"s);

	shaders.font_vs = create_vs(R"(
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
	shaders.font_ps = create_ps(HLSL_CBUFFER R"(
Texture2D tex : register(t0);
void main(in float2 uv : UV, out float4 pixel_color0 : SV_Target, out float4 pixel_color1 : SV_Target1) {
	pixel_color0 = 1;
	pixel_color1 = float4(tex.Sample(default_sampler, uv).rgb, 1);
}
)"s);

	shaders.crosshair_vs = create_vs(R"(
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
	shaders.crosshair_ps = create_ps(R"(
void main(out float4 pixel_color : SV_Target) {
	pixel_color = 1;
}
)"s);

	shaders.grass_vs = create_vs(HLSL_CBUFFER HLSL_COMMON R"(
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

	shadow_map_uv = mul(light_vp_matrix, float4(pos, 1));
}
)"s);

	shaders.grass_ps = create_ps(HLSL_CBUFFER HLSL_COMMON HLSL_LIGHTING HLSL_TRIPLANAR R"(
void main(
	in float2 uv : UV,
	in float3 normal : NORMAL,
	in float3 wpos : WPOS,
	in float3 view : VIEW,
	in float4 shadow_map_uv : SHADOW_MAP_UV,
	in float4 pixel_position : SV_Position,

	out float4 pixel_color : SV_Target
) {
	float4 colortex = albedo_texture.Sample(default_sampler, uv);

	clip(colortex.a-0.5f);

	float3 albedo = colortex.rgb * float3(.29,.69,.14);

	float3 N = normal;

	float roughness = 1;

	pixel_color = surface(N, c_ldir, view, shadow_map_uv, albedo, 1, roughness, pixel_position);
}
)"s);

	shaders.chunk_block_vs = create_vs(HLSL_CBUFFER HLSL_COMMON R"(

struct ChunkVertex {
	float3 position;
	uint normal;
	float3 parent_position;
	uint parent_normal;
};

StructuredBuffer<ChunkVertex> s_vertex_buffer : VERTEX_BUFFER_SLOT;

void main(
	in uint vertex_id : SV_VertexID,

	out float3 normal : NORMAL,
	out float3 wpos : WPOS,
	out float3 view : VIEW,
	out float4 shadow_map_uv : SHADOW_MAP_UV,
	out float4 position : SV_Position
) {
	ChunkVertex vertex = s_vertex_buffer[vertex_id];
	float3 pos = vertex.position + c_relative_position;
	wpos = pos;
	normal = decode_normal(vertex.normal);
	view = pos - c_campos;

	position = mul(c_mvp, float4(pos, 1.0f));

	shadow_map_uv = mul(light_vp_matrix, float4(pos, 1));
}
)"s);

	shaders.chunk_block_ps = create_ps(HLSL_CBUFFER HLSL_COMMON HLSL_LIGHTING HLSL_TRIPLANAR R"(
void main(
	in float3 normal : NORMAL,
	in float3 wpos : WPOS,
	in float3 view : VIEW,
	in float4 shadow_map_uv : SHADOW_MAP_UV,
	in float4 pixel_position : SV_Position,

	out float4 pixel_color : SV_Target
) {
	normal = normalize(normal);

	float3 N = float4(triplanar_normal(normal_texture, default_sampler, wpos, normal), 1);

	float3 albedo = triplanar(albedo_texture, default_sampler, wpos, normal).xyz;

	float metalness = 0;
	float roughness = 1;

	pixel_color = surface(N, c_ldir, view, shadow_map_uv, albedo, 1, roughness, pixel_position);
}
)"s);

	shaders.tree_vs = create_vs(HLSL_CBUFFER R"(
struct Vertex {
	float3 position;
	float3 normal;
	float2 uv;
	float4 tangent;
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
	out float3 tangent : TANGENT,
	out float2 uv : UV,
	out float3 wpos : WPOS,
	out float3 view : VIEW,
	out float4 shadow_map_uv : SHADOW_MAP_UV,
	out float4 position : SV_Position
) {
	Vertex vertex = s_vertex_buffer[vertex_id+c_vertex_offset];
	Instance instance = s_instance_buffer[instance_id];
	float3 pos = mul(instance.mat, vertex.position) + instance.position + c_relative_position;

	wpos = pos;
	normal = mul(instance.mat, float4(vertex.normal, 0)).xyz;

	tangent = mul(instance.mat, float4(vertex.tangent.xyz, 0)).xyz * vertex.tangent.w;
	//tangent.y *= -1;
	//tangent.xyz *= -1;

	view = pos - c_campos;

	uv = vertex.uv;

	position = mul(c_mvp, float4(pos, 1.0f));

	shadow_map_uv = mul(light_vp_matrix, float4(pos, 1));
}
)"s);

	shaders.tree_ps = create_ps(HLSL_CBUFFER HLSL_COMMON HLSL_LIGHTING HLSL_TRIPLANAR R"(
void main(
	in float3 normal : NORMAL,
	in float3 tangent : TANGENT,
	in float2 uv : UV,
	in float3 wpos : WPOS,
	in float3 view : VIEW,
	in float4 shadow_map_uv : SHADOW_MAP_UV,
	in float4 pixel_position : SV_Position,

	out float4 pixel_color : SV_Target
) {
	float lod_mask = lod_mask_texture.Load(int3(int2(pixel_position.xy) & 63, 0)).x;
	float4 data = albedo_texture.Sample(default_sampler, uv);
	clip(min(data.a - .5, sign(c_lod_t) * (abs(c_lod_t) - lod_mask)));

	normal = normalize(normal);
	float3 albedo = data.rgb;
	pixel_color.a = data.a;

	float3 tsn = normal_texture.Sample(default_sampler, uv).rgb;
	float3 N = world_normal(normal, tangent, tsn, 1);

	float ao = ao_texture.Sample(default_sampler, uv).x;

	float metalness = 0;
	float roughness = 1;

	pixel_color = surface(N, c_ldir, view, shadow_map_uv, albedo, ao, roughness, pixel_position);
}
)"s);

	shaders.tree_shadow_vs = create_vs(HLSL_CBUFFER R"(
struct Vertex {
	float3 position;
	float3 normal;
	float2 uv;
	float4 tangent;
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
	out float4 position : SV_Position
) {
	Vertex vertex = s_vertex_buffer[vertex_id+c_vertex_offset];
	Instance instance = s_instance_buffer[instance_id];
	float3 pos = mul(instance.mat, vertex.position) + instance.position + c_relative_position;
	position = mul(c_mvp, float4(pos, 1.0f));
	uv = vertex.uv;
}
)"s);

	shaders.tree_shadow_ps = create_ps(HLSL_CBUFFER HLSL_COMMON HLSL_LIGHTING HLSL_TRIPLANAR R"(
void main(
	in float2 uv : UV,
	in float4 pixel_position : SV_Position
) {
	//float lod_mask = lod_mask_texture.Load(int3(int2(pixel_position.xy) & 63, 0)).x;
	//clip(sign(c_lod_t) * (abs(c_lod_t) - lod_mask));

	float4 data = albedo_texture.Sample(default_sampler, uv);
	clip(data.a - 0.5);
}
)"s);
	shaders.collider_vs = create_vs(HLSL_CBUFFER HLSL_COMMON R"(
struct Vertex {
	float3 position;
	uint color;
};

StructuredBuffer<Vertex> s_vertex_buffer : VERTEX_BUFFER_SLOT;

void main(
	in uint vertex_id : SV_VertexID,
	out float4 position : SV_Position
) {
	float3 pos = s_vertex_buffer[vertex_id].position;

	position = mul(c_mvp, float4(pos, 1.0f));
}
)"s);
	shaders.collider_ps = create_ps(R"(
void main(out float4 pixel_color : SV_Target) {
	pixel_color = float4(0,1,0,1);
}
)"s);

	shaders.color_vs = create_vs(R"(
void main(
	in uint vertex_id : SV_VertexID,
	out float4 position : SV_Position
) {
	float2 positions[] = {
		{-1,-1}, {-1, 1}, { 1,-1},
		{-1, 1}, { 1, 1}, { 1,-1},
	};
	position = float4(positions[vertex_id], 0, 1);
}
)"s);
	shaders.color_ps = create_ps(HLSL_CBUFFER R"(
void main(out float4 pixel_color : SV_Target) {
	pixel_color = c_random;
}
)"s);


	shaders.model_vs = create_vs(HLSL_CBUFFER R"(
struct Vertex {
	float3 position;
	float3 normal;
	float2 uv;
	float4 tangent;
};

StructuredBuffer<Vertex> s_vertex_buffer : VERTEX_BUFFER_SLOT;

void main(
	in uint vertex_id : SV_VertexID,

	out float3 normal : NORMAL,
	out float3 tangent : TANGENT,
	out float2 uv : UV,
	out float3 wpos : WPOS,
	out float3 view : VIEW,
	out float4 shadow_map_uv : SHADOW_MAP_UV,
	out float4 position : SV_Position
) {
	Vertex vertex = s_vertex_buffer[vertex_id+c_vertex_offset];
	float3 pos = mul(c_model, vertex.position) + c_relative_position;

	wpos = pos;
	normal = mul(c_model, float4(vertex.normal, 0)).xyz;

	tangent = mul(c_model, float4(vertex.tangent.xyz, 0)).xyz * vertex.tangent.w;
	//tangent.y *= -1;
	//tangent.xyz *= -1;

	view = pos - c_campos;

	uv = vertex.uv;

	position = mul(c_mvp, float4(pos, 1.0f));

	shadow_map_uv = mul(light_vp_matrix, float4(pos, 1));
}
)"s);

	shaders.model_ps = create_ps(HLSL_CBUFFER HLSL_COMMON HLSL_LIGHTING HLSL_TRIPLANAR R"(
void main(
	in float3 normal : NORMAL,
	in float3 tangent : TANGENT,
	in float2 uv : UV,
	in float3 wpos : WPOS,
	in float3 view : VIEW,
	in float4 shadow_map_uv : SHADOW_MAP_UV,
	in float4 pixel_position : SV_Position,

	out float4 pixel_color : SV_Target
) {
	float lod_mask = lod_mask_texture.Load(int3(int2(pixel_position.xy) & 63, 0)).x;
	float4 data = albedo_texture.Sample(default_sampler, uv);
	//clip(min(data.a - .5, sign(c_lod_t) * (abs(c_lod_t) - lod_mask)));
	clip(data.a - .5);

	normal = normalize(normal);
	float3 albedo = data.rgb;
	pixel_color.a = data.a;

	float3 tsn = normal_texture.Sample(default_sampler, uv).rgb;
	float3 N = world_normal(normal, tangent, tsn, 1);

	float ao = ao_texture.Sample(default_sampler, uv).x;

	float metalness = 0;
	float roughness = 1;

	pixel_color = surface(N, c_ldir, view, shadow_map_uv, albedo, ao, roughness, pixel_position);
}
)"s);
	shaders.model_shadow_vs = create_vs(HLSL_CBUFFER R"(
struct Vertex {
	float3 position;
	float3 normal;
	float2 uv;
	float4 tangent;
};

StructuredBuffer<Vertex> s_vertex_buffer : VERTEX_BUFFER_SLOT;

void main(
	in uint vertex_id : SV_VertexID,
	in uint instance_id : SV_InstanceID,
	out float2 uv : UV,
	out float4 position : SV_Position
) {
	Vertex vertex = s_vertex_buffer[vertex_id+c_vertex_offset];
	float3 pos = mul(c_model, vertex.position) + c_relative_position;
	position = mul(c_mvp, float4(pos, 1.0f));
	uv = vertex.uv;
}
)"s);

	shaders.model_shadow_ps = create_ps(HLSL_CBUFFER HLSL_COMMON HLSL_LIGHTING HLSL_TRIPLANAR R"(
void main(
	in float2 uv : UV,
	in float4 pixel_position : SV_Position
) {
	//float lod_mask = lod_mask_texture.Load(int3(int2(pixel_position.xy) & 63, 0)).x;
	//clip(sign(c_lod_t) * (abs(c_lod_t) - lod_mask));

	float4 data = albedo_texture.Sample(default_sampler, uv);
	clip(data.a - 0.5);
}
)"s);
}
