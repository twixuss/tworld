#pragma once
#include "d3d11.h"

struct Shaders {
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

    ID3D11VertexShader *collider_vs = 0;
    ID3D11PixelShader  *collider_ps = 0;

    ID3D11VertexShader *color_vs = 0;
    ID3D11PixelShader  *color_ps = 0;

    ID3D11VertexShader *model_vs = 0;
    ID3D11PixelShader  *model_ps = 0;

    ID3D11VertexShader *model_shadow_vs = 0;
    ID3D11PixelShader  *model_shadow_ps = 0;
};

extern Shaders shaders;
