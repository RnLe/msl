let wasmModule: any = null;
let isInitialized = false;

export async function initWasm() {
  if (isInitialized && wasmModule) return wasmModule;
  
  try {
    // Import the WASM module (web target) from public directory
    const wasm = await import("../../public/wasm/moire_lattice_wasm.js");
    
    // For web target, the default export should be the init function
    if (wasm.default && typeof wasm.default === 'function') {
      // For static sites, we need to provide the WASM file path
      await wasm.default('/wasm/moire_lattice_wasm_bg.wasm');
    }
    
    wasmModule = wasm;
    isInitialized = true;
    return wasm;
  } catch (error) {
    console.error('Failed to initialize WASM:', error);
    throw error;
  }
}

export async function getWasmModule() {
  return await initWasm();
}

// Utility function to check if WASM is loaded
export function isWasmLoaded(): boolean {
  return wasmModule !== null && isInitialized;
}
