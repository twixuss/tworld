#include <tl/math.h>
#include <tl/array.h>
#include "common.h"

using namespace tl;

template <class Fn>
constexpr void iterate_grid_from_center(s32 radius, Fn &&fn) {
	fn(v3s{});
	for (s32 i = 0; i < radius; ++i) {
		for (s32 j = -1; j <= 1; j += 2) {
			for (s32 y = i; y >= -i; --y) {
			for (s32 z = i; z >= -i; --z) {
				fn(v3s{(i+1)*j, y, z});
			}
			}
		}
		for (s32 j = -1; j <= 1; j += 2) {
			for (s32 x = i+1; x >= -i-1; --x) {
			for (s32 z = i; z >= -i; --z) {
				fn(v3s{x, (i+1)*j, z});
			}
			}
		}
		for (s32 j = -1; j <= 1; j += 2) {
			for (s32 y = i+1; y >= -i-1; --y) {
			for (s32 x = i+1; x >= -i-1; --x) {
				fn(v3s{x, y, (i+1)*j});
			}
			}
		}
	}

}

extern "C" const Array<v3s8, pow3(DRAWD*2+1)> grid_map = [] {
	Array<v3s8, pow3(DRAWD*2+1)> r = {};
	u32 i = 0;
	iterate_grid_from_center(DRAWD, [&](v3s v) {
		r[i++] = (v3s8)v;
	});
	return r;
}();
