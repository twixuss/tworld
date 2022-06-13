#pragma once
#include <d3d11.h>

extern IDXGISwapChain *swap_chain;
extern ID3D11Device *device;
extern ID3D11DeviceContext *immediate_context;
extern ID3D11RenderTargetView *back_buffer;
extern ID3D11InfoQueue* debug_info_queue;

inline void print_messages() {
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
