#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;
    
    wasm_bindgen_test_configure!(run_in_browser);
    
    #[wasm_bindgen_test]
    fn test_hexagonal_brillouin_zone_vertices() {
        // Create hexagonal lattice
        let lattice = create_hexagonal_lattice(1.0).unwrap();
        
        // Get WS cell
        let ws_cell = lattice.wigner_seitz_cell();
        let ws_data = ws_cell.get_data().unwrap();
        
        // Get BZ
        let bz = lattice.brillouin_zone();
        let bz_data = bz.get_data().unwrap();
        
        // Debug print to console (will appear in test output)
        web_sys::console::log_1(&format!("WS cell data: {:?}", ws_data).into());
        web_sys::console::log_1(&format!("BZ data: {:?}", bz_data).into());
        
        // Deserialize the data back to check structure
        let ws_parsed: PolyhedronData = serde_wasm_bindgen::from_value(ws_data).unwrap();
        let bz_parsed: PolyhedronData = serde_wasm_bindgen::from_value(bz_data).unwrap();
        
        web_sys::console::log_1(&format!("WS vertices count: {}", ws_parsed.vertices.len()).into());
        web_sys::console::log_1(&format!("BZ vertices count: {}", bz_parsed.vertices.len()).into());
        
        // Both should have 6 vertices for hexagonal lattice
        assert_eq!(ws_parsed.vertices.len(), 6, "WS cell should have 6 vertices");
        assert_eq!(bz_parsed.vertices.len(), 6, "BZ should have 6 vertices");
        
        // Log the actual vertex coordinates
        for (i, vertex) in bz_parsed.vertices.iter().enumerate() {
            web_sys::console::log_1(&format!("BZ vertex {}: ({:.6}, {:.6}, {:.6})", i, vertex.x, vertex.y, vertex.z).into());
        }
    }
}
