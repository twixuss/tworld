#pragma once
#include "d3d11.h"
#include <tl/common.h>

using namespace tl;

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
