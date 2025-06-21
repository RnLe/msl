let wasmModule: any = null;
let loadingPromise: Promise<any> | null = null;
let isInitialized = false;

export async function initWasm() {
  if (isInitialized) return wasmModule;
  
  const wasm = await getWasmModule();
  // Initialize the WASM module if needed
  if (wasm.default && typeof wasm.default === 'function') {
    await wasm.default();
  }
  isInitialized = true;
  return wasm;
}

export async function getWasmModule() {
  if (wasmModule) return wasmModule;
  
  if (loadingPromise) return loadingPromise;
  
  loadingPromise = import("../../wasm/wasm.js").then((wasm) => {
    wasmModule = wasm;
    return wasm;
  });
  
  return loadingPromise;
}

// Utility function to check if WASM is loaded
export function isWasmLoaded(): boolean {
  return wasmModule !== null && isInitialized;
}
