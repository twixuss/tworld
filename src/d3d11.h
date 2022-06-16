#pragma once
#include <d3d11.h>

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
    }
}

template <umm w, umm h>
inline ID3D11ShaderResourceView *make_texture(v4u8 (&pixels)[w][h]) {
	ID3D11Texture2D *resource;
	{
		D3D11_TEXTURE2D_DESC desc {
			.Width = w,
			.Height = h,
			.MipLevels = log2(ceil_to_power_of_2(max(w,h)))+1,
			.ArraySize = 1,
			.Format = DXGI_FORMAT_R8G8B8A8_UNORM,
			.SampleDesc = {1,0},
			.BindFlags = D3D11_BIND_SHADER_RESOURCE|D3D11_BIND_RENDER_TARGET,
			.MiscFlags = D3D11_RESOURCE_MISC_GENERATE_MIPS,
		};

		dhr(device->CreateTexture2D(&desc, 0, &resource));
		immediate_context->UpdateSubresource(resource, 0, 0, pixels, w*sizeof(v4u8), 1);
	}

	ID3D11ShaderResourceView *view;
	{
		D3D11_SHADER_RESOURCE_VIEW_DESC desc {
			.Format = DXGI_FORMAT_R8G8B8A8_UNORM,
			.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D,
			.Texture2D = {
				.MipLevels = (UINT)-1,
			},
		};
		dhr(device->CreateShaderResourceView(resource, &desc, &view));
	}

	immediate_context->GenerateMips(view);

	return view;
}
