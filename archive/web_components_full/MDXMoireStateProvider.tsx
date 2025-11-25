'use client';

import { createContext, useContext, useState, ReactNode } from 'react';
import type { WasmMoire2D, WasmLattice2D } from '../../public/wasm/moire_lattice_wasm';

type MoireLatticeState = {
  moireLattice: WasmMoire2D | null;
  baseLattice: WasmLattice2D | null;
  setMoireLattice: (lattice: WasmMoire2D | null) => void;
  setBaseLattice: (lattice: WasmLattice2D | null) => void;
};

const MoireLatticeContext = createContext<MoireLatticeState | null>(null);

export function MDXMoireStateProvider({ children }: { children: ReactNode }) {
  const [moireLattice, setMoireLattice] = useState<WasmMoire2D | null>(null);
  const [baseLattice, setBaseLattice] = useState<WasmLattice2D | null>(null);

  return (
    <MoireLatticeContext.Provider 
      value={{ 
        moireLattice, 
        baseLattice, 
        setMoireLattice, 
        setBaseLattice 
      }}
    >
      {children}
    </MoireLatticeContext.Provider>
  );
}

// Handy hook for components that need access to the moiré lattice state
export function useMoireLatticeState() {
  const context = useContext(MoireLatticeContext);
  if (!context) {
    throw new Error('useMoireLatticeState must be used within MDXMoireStateProvider');
  }
  return context;
}
