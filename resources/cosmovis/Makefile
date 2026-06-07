# Everything runs inside the cosmopower micromamba env so the project
# has one consistent toolchain (emscripten + nodejs were added to it).
MM      := micromamba run -n cosmopower
SRC     := src/cosmo_emu.c
WASM_JS := web/cosmo_emu.js
WASM    := web/cosmo_emu.wasm

EMCC_FLAGS := -O3 \
    -s MODULARIZE=1 \
    -s EXPORT_NAME=CosmoEmuModule \
    -s EXPORT_ES6=1 \
    -s ENVIRONMENT=web,worker,node \
    -s ALLOW_MEMORY_GROWTH=1 \
    -s INITIAL_MEMORY=16MB \
    -s EXPORTED_FUNCTIONS='["_cp_load","_cp_predict","_cp_free","_cp_malloc","_cp_mfree","_cp_n_outputs","_cp_output_kind","_cp_ell_min","_cp_ell_max","_cp_n_inputs","_sht_synth"]' \
    -s EXPORTED_RUNTIME_METHODS='["HEAPU8","HEAPF32","HEAP32"]'

.PHONY: all build weights wasm verify serve clean

all: wasm

build: weights wasm

weights:
	$(MM) python tools/export_weights.py

wasm: $(WASM_JS)

$(WASM_JS): $(SRC)
	$(MM) emcc $(SRC) $(EMCC_FLAGS) -o $(WASM_JS)

verify: build
	$(MM) python tools/spot_check.py

serve:
	cd web && $(MM) python -m http.server 6011

clean:
	rm -f $(WASM_JS) $(WASM)
	rm -rf web/weights
