#pragma once
#include <d3d11.h>

#include <tl/common.h>
using namespace tl;

extern IDXGISwapChain *swap_chain;
extern ID3D11Device *device;
extern ID3D11DeviceContext *immediate_context;
extern ID3D11RenderTargetView *back_buffer;
extern ID3D11InfoQueue* debug_info_queue;

inline void print_messages() {
    if (!debug_info_queue)
        return;
    UINT64 message_count = debug_info_queue->GetNumStoredMessages();

    for(UINT64 i = 0; i < message_count; i++){
        SIZE_T message_size = 0;
        debug_info_queue->GetMessage(i, nullptr, &message_size); //get the size of the message

        D3D11_MESSAGE* message = (D3D11_MESSAGE*) malloc(message_size); //allocate enough space
        debug_info_queue->GetMessage(i, message, &message_size); //get the actual message

        //do whatever you want to do with it
        print("D3D11: {}\n", Span((char *)message->pDescription, message->DescriptionByteLength));

        free(message);
    }

    debug_info_queue->ClearStoredMessages();
}

inline void dhr(HRESULT hr, std::source_location location = std::source_location::current()) {
    if (FAILED(hr)) {
        print_messages();
        print("bad HRESULT at {}\n", location);
        debug_break();
    } else {
        print_messages();
	}
}

inline ID3D11ShaderResourceView *make_texture(void *pixels, u32 w, u32 h, u32 pixel_size, DXGI_FORMAT format, bool mips = true) {
	ID3D11Texture2D *resource;
	defer { resource->Release(); };

	{
		D3D11_TEXTURE2D_DESC desc {
			.Width = w,
			.Height = h,
			.MipLevels = mips ? log2(ceil_to_power_of_2(max(w,h)))+1 : 1,
			.ArraySize = 1,
			.Format = format,
			.SampleDesc = {1,0},
			.BindFlags = (UINT)(D3D11_BIND_SHADER_RESOURCE | (mips?D3D11_BIND_RENDER_TARGET:0)),
			.MiscFlags = (UINT)(mips?D3D11_RESOURCE_MISC_GENERATE_MIPS:0),
		};

		dhr(device->CreateTexture2D(&desc, 0, &resource));
		immediate_context->UpdateSubresource(resource, 0, 0, pixels, w*pixel_size, 1);
	}

	ID3D11ShaderResourceView *view;
	{
		D3D11_SHADER_RESOURCE_VIEW_DESC desc {
			.Format = format,
			.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D,
			.Texture2D = {
				.MipLevels = (UINT)-1,
			},
		};
		dhr(device->CreateShaderResourceView(resource, &desc, &view));
	}

	if (mips)
		immediate_context->GenerateMips(view);

	return view;
}
inline ID3D11ShaderResourceView *make_texture(void *pixels, u32 w, u32 h, u32 d, u32 pixel_size, DXGI_FORMAT format, bool mips = true) {
	ID3D11Texture3D *resource;
	defer { resource->Release(); };

	{
		D3D11_TEXTURE3D_DESC desc {
			.Width = w,
			.Height = h,
			.Depth = d,
			.MipLevels = mips ? log2(ceil_to_power_of_2(max(w,h)))+1 : 1,
			.Format = format,
			.BindFlags = (UINT)(D3D11_BIND_SHADER_RESOURCE | (mips?D3D11_BIND_RENDER_TARGET:0)),
			.MiscFlags = (UINT)(mips?D3D11_RESOURCE_MISC_GENERATE_MIPS:0),
		};

		dhr(device->CreateTexture3D(&desc, 0, &resource));
		immediate_context->UpdateSubresource(resource, 0, 0, pixels, w*pixel_size, w*h*pixel_size);
	}

	ID3D11ShaderResourceView *view;
	{
		D3D11_SHADER_RESOURCE_VIEW_DESC desc {
			.Format = format,
			.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE3D,
			.Texture3D = {
				.MipLevels = (UINT)-1,
			},
		};
		dhr(device->CreateShaderResourceView(resource, &desc, &view));
	}

	if (mips)
		immediate_context->GenerateMips(view);

	return view;
}
inline ID3D11ShaderResourceView *make_texture(void *pixels, u32 w, u32 h, bool mips = true) {
	return make_texture(pixels, w, h, 4, DXGI_FORMAT_R8G8B8A8_UNORM, mips);
}
template <umm w, umm h>
inline ID3D11ShaderResourceView *make_texture(v4u8 (&pixels)[w][h], bool mips = true) {
	return make_texture(pixels, w, h, 4, DXGI_FORMAT_R8G8B8A8_UNORM, mips);
}
template <umm w, umm h>
inline ID3D11ShaderResourceView *make_texture(u8 (&pixels)[w][h], bool mips = true) {
	return make_texture(pixels, w, h, 1, DXGI_FORMAT_R8_UNORM, mips);
}
template <umm w, umm h, umm d>
inline ID3D11ShaderResourceView *make_texture(u8 (&pixels)[w][h][d], bool mips = true) {
	return make_texture(pixels, w, h, d, 1, DXGI_FORMAT_R8_UNORM, mips);
}

template <class T>
ID3D11ShaderResourceView *create_structured_buffer(Span<T> elements) {
	ID3D11Buffer *resource;
	defer { resource->Release(); };

	{
		D3D11_BUFFER_DESC desc {
			.ByteWidth = (UINT)(sizeof(elements[0]) * elements.count),
			.Usage = D3D11_USAGE_DEFAULT,
			.BindFlags = D3D11_BIND_SHADER_RESOURCE,
			.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED,
			.StructureByteStride = sizeof(elements[0]),
		};
		D3D11_SUBRESOURCE_DATA init {
			.pSysMem = elements.data,
		};
		dhr(device->CreateBuffer(&desc, &init, &resource));
	}
	ID3D11ShaderResourceView *srv;
	dhr(device->CreateShaderResourceView(resource, 0, &srv));
	return srv;
}

template <class T>
void update_structured_buffer(ID3D11ShaderResourceView *srv, Span<T> elements) {
	ID3D11Resource *buffer;
	srv->GetResource(&buffer);
	defer { buffer->Release(); };

	D3D11_BOX box {
		.left = 0,
		.top = 0,
		.front = 0,
		.right = (UINT)(elements.count * sizeof(T)),
		.bottom = 1,
		.back = 1,
	};

	immediate_context->UpdateSubresource(buffer, 0, &box, elements.data, 0, 0);
}

template <class T>
ID3D11Buffer *create_index_buffer(Span<T> indices) {
	D3D11_BUFFER_DESC desc {
		.ByteWidth = (UINT)(sizeof(indices[0]) * indices.count),
		.Usage = D3D11_USAGE_DEFAULT,
		.BindFlags = D3D11_BIND_INDEX_BUFFER,
		.StructureByteStride = sizeof(indices[0]),
	};
	D3D11_SUBRESOURCE_DATA init {
		.pSysMem = indices.data,
	};

	ID3D11Buffer *buffer;
	dhr(device->CreateBuffer(&desc, &init, &buffer));
	return buffer;
}

template <class T>
struct SBuffer {
	ID3D11ShaderResourceView *srv = 0;
	u32 count;

	void update(Span<T> elements) {
		if (srv) {
			if (elements.count > count) {
				srv->Release();
				srv = create_structured_buffer(elements);
			} else {
				update_structured_buffer(srv, elements);
			}
		} else {
			srv = create_structured_buffer(elements);
		}
		count = elements.count;
	}
	friend void free(SBuffer &buffer) {
		if (buffer.srv) {
			buffer.srv->Release();
			buffer.srv = 0;
		}
		buffer.count = 0;
	}
};

template <class T>
umm append(StringBuilder &builder, SBuffer<T> v) {
	return append_format(builder, "{{srv={}, count={}}}", v.srv, v.count);
}

template <class T = void>
struct CBuffer;

struct CBufferBase {
    ID3D11Buffer *cbuffer;

    static void init(CBufferBase &buffer, u32 size) {
        D3D11_BUFFER_DESC desc {
            .ByteWidth = size,
            .Usage = D3D11_USAGE_DEFAULT,
            .BindFlags = D3D11_BIND_CONSTANT_BUFFER,
        };
        dhr(device->CreateBuffer(&desc, 0, &buffer.cbuffer));
    }

    static CBuffer<> create(u32 size);

    template <class T>
    static CBuffer<T> create();
};

template <class T = void>
struct CBuffer : CBufferBase {
    void init() {
        CBufferBase::init(*this, sizeof(T));
    }

    void update(T const &data) {
        immediate_context->UpdateSubresource(cbuffer, 0, 0, &data, 0, 0);
    }
};

template <>
struct CBuffer<void> : CBufferBase {
    void update(void const *data) {
        immediate_context->UpdateSubresource(cbuffer, 0, 0, data, 0, 0);
    }
};

CBuffer<> CBufferBase::create(u32 size) {
    CBuffer<> result;
    init(result, size);
    return result;
}

template <class T>
CBuffer<T> CBufferBase::create() {
    CBuffer<T> result;
    init(result, sizeof(T));
    return result;
}

struct alignas(16) FrameCbuffer {
    m4 mvp;
    m4 rotproj;
	m4 light_vp_matrix;
	v3f campos;
	f32 time;
	v3f ldir;
	f32 frame;
};
CBuffer<FrameCbuffer> frame_cbuffer;

struct alignas(16) ChunkCbuffer {
    v3f relative_position;
	f32 was_remeshed;
    v3f actual_position;
	u32 vertex_offset;
	f32 lod_t;
};

CBuffer<ChunkCbuffer> chunk_cbuffer;

#define DEFAULT_SAMPLER_SLOT       0
#define DEFAULT_NOMIP_SAMPLER_SLOT 1
#define NEAREST_SAMPLER_SLOT       2
#define SHADOW_SAMPLER_SLOT        3

#define VERTEX_BUFFER_SLOT 0
#define INSTANCE_BUFFER_SLOT 1

#define ALBEDO_TEXTURE_SLOT   2
#define NORMAL_TEXTURE_SLOT   3
#define AO_TEXTURE_SLOT       4
#define SKY_TEXTURE_SLOT      5
#define SHADOW_TEXTURE_SLOT   6
#define LOD_MASK_TEXTURE_SLOT 7

#define HLSL_CBUFFER R"(
cbuffer _ : register(b0) {
    float4x4 c_mvp;
    float4x4 c_rotproj;
	float4x4 light_vp_matrix;
	float3 c_campos;
	float c_time;
	float3 c_ldir;
	float c_frame;
}
cbuffer _ : register(b1) {
    float3 c_relative_position;
	float c_was_remeshed;
    float3 c_actual_position;
	uint c_vertex_offset;
	float c_lod_t;
}

SamplerState default_sampler          : register(s)" STRINGIZE(DEFAULT_SAMPLER_SLOT) R"();
SamplerState default_nomip_sampler    : register(s)" STRINGIZE(DEFAULT_NOMIP_SAMPLER_SLOT) R"();
SamplerState nearest_sampler          : register(s)" STRINGIZE(NEAREST_SAMPLER_SLOT) R"();
SamplerComparisonState shadow_sampler : register(s)" STRINGIZE(SHADOW_SAMPLER_SLOT) R"();

#define VERTEX_BUFFER_SLOT   register(t)" STRINGIZE(VERTEX_BUFFER_SLOT) R"()
#define INSTANCE_BUFFER_SLOT register(t)" STRINGIZE(INSTANCE_BUFFER_SLOT) R"()
Texture2D albedo_texture   : register(t)" STRINGIZE(ALBEDO_TEXTURE_SLOT) R"();
Texture2D normal_texture   : register(t)" STRINGIZE(NORMAL_TEXTURE_SLOT) R"();
Texture2D ao_texture       : register(t)" STRINGIZE(AO_TEXTURE_SLOT) R"();
Texture2D sky_texture      : register(t)" STRINGIZE(SKY_TEXTURE_SLOT) R"();
Texture2D shadow_texture   : register(t)" STRINGIZE(SHADOW_TEXTURE_SLOT) R"();
Texture3D lod_mask_texture : register(t)" STRINGIZE(LOD_MASK_TEXTURE_SLOT) R"();

)"

#define HLSL_COMMON R"(
#define X(t) \
t pow2(t v){return v*v;} \
t pow3(t v){return v*v*v;} \
t pow4(t v){return pow2(v*v);} \
t pow5(t v){return pow2(v*v)*v;}
X(float)
X(float2)
X(float3)
X(float4)
#undef X

float sdot(float3 a, float3 b) {
	return max(0,dot(a,b));
}

#define X(t) \
t map(t value, t source_min, t source_max, t dest_min, t dest_max) { \
	return (value - source_min) / (source_max - source_min) * (dest_max - dest_min) + dest_min; \
}
X(float)
X(float2)
X(float3)
X(float4)
#undef X

#define X(t) \
t map_clamped(t value, t source_min, t source_max, t dest_min, t dest_max) { \
	return (clamp(value, min(source_min, source_max), max(source_min, source_max)) - source_min) / (source_max - source_min) * (dest_max - dest_min) + dest_min; \
}
X(float)
X(float2)
X(float3)
X(float4)
#undef X

#define pi 3.1415926535897932384626433832795

#define CHUNKW )" STRINGIZE(CHUNKW) R"(
#define DRAWD )" STRINGIZE(DRAWD) R"(
#define FARD )" STRINGIZE(FARD) R"(

float3 srgb_to_rgb(float3 i)
{
    //if (i.x <= 0.04045)
	//	i.x /= 12.92;
	//else
	//	i.x = pow((i.x + 0.055) / 1.055, 2.4);

	return lerp(i / 12.92, pow((i + 0.055) / 1.055, 2.4), i > 0.04045);
}
float3 rgb_to_srgb(float3 i)
{
	return lerp(i * 12.92, 1.055 * pow(i, 1.0 / 2.4) - 0.055, i > 0.0031308);
}

uint encode_normal(float3 n) {
	float e = 0.001;
	return
		((uint)map(n.x, -1, 1, 0, 256-e) << 0) |
		((uint)map(n.y, -1, 1, 0, 256-e) << 8) |
		((uint)map(n.z, -1, 1, 0, 256-e) << 16);
}

float3 decode_normal(uint n) {
	return float3(
		map((n >>  0) & 0xff, 0, 256, -1, 1),
		map((n >>  8) & 0xff, 0, 256, -1, 1),
		map((n >> 16) & 0xff, 0, 256, -1, 1)
	);
}

)"

#define HLSL_LIGHTING R"(
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


float3 calculate_normal(float3 normal, float3 tangent, float3 tangent_space_normal)
{
	return normalize(
		tangent_space_normal.x * tangent +
		tangent_space_normal.y * cross(normal, tangent) +
		tangent_space_normal.z * normal
	);
}
float3 calculate_normal(float3 normal, float4 tangent, float3 tangent_space_normal)
{
	return normalize(
		tangent_space_normal.x * tangent.xyz +
		tangent_space_normal.y * cross(normal, tangent.xyz) * tangent.w +
		tangent_space_normal.z * normal
	);
}
float3 unpack_normal(float3 color, float scale) {
	return map(
		color,
		0,
		1,
		float3(-scale,-scale,0),
		float3(+scale,+scale,1)
	);
}
float3 world_normal(float3 normal, float3 tangent, float3 color, float scale) {
	return calculate_normal(normal, tangent, unpack_normal(color, scale));
}
float3 world_normal(float3 normal, float4 tangent, float3 color, float scale) {
	return calculate_normal(normal, tangent, unpack_normal(color, scale));
}
)"

#define HLSL_TRIPLANAR R"(
float4 triplanar(Texture2D tex, SamplerState sam, float3 wpos, float3 normal) {
	float4 pixel_color = 0;

	float3 t = abs(normal);
	// t = pow(t, 16);
	t *= 1.0f / (t.x + t.y + t.z);

	float2 u = wpos.zy * float2(+sign(normal.x), 1);
	float2 v = wpos.xz * float2(+sign(normal.y), 1);
	float2 w = wpos.xy * float2(-sign(normal.z), 1);

	pixel_color += tex.Sample(sam, u) * t.x;
	pixel_color += tex.Sample(sam, v) * t.y;
	pixel_color += tex.Sample(sam, w) * t.z;

	return pixel_color;
}

// I tried to make an overcomplicated mess as always and it didn't work.
// So i'm using an algorithm described in here:
// https://bgolus.medium.com/normal-mapping-for-a-triplanar-shader-10bf39dca05a
// They are also describing different blending techniques, but i think this is already good enough.
float3 triplanar_normal(Texture2D tex, SamplerState sam, float3 wpos, float3 normal) {
	float3 t = abs(normal);
	// t = pow(t, 16);
	t *= 1.0f / (t.x + t.y + t.z);

	float normal_scale = .5;
	float3 nx = unpack_normal(tex.Sample(sam, wpos.zy).rgb, normal_scale);
	float3 ny = unpack_normal(tex.Sample(sam, wpos.xz).rgb, normal_scale);
	float3 nz = unpack_normal(tex.Sample(sam, wpos.xy).rgb, normal_scale);
	nx.z *= sign(normal).x;
	ny.z *= sign(normal).y;
	nz.z *= sign(normal).z;

	return normalize(nx.zyx * t.x + ny.xzy * t.y + nz.xyz * t.z);
}
)"
