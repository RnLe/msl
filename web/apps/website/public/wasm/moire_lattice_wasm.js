let wasm;

let WASM_VECTOR_LEN = 0;

let cachedUint8ArrayMemory0 = null;

function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

const cachedTextEncoder = (typeof TextEncoder !== 'undefined' ? new TextEncoder('utf-8') : { encode: () => { throw Error('TextEncoder not available') } } );

const encodeString = (typeof cachedTextEncoder.encodeInto === 'function'
    ? function (arg, view) {
    return cachedTextEncoder.encodeInto(arg, view);
}
    : function (arg, view) {
    const buf = cachedTextEncoder.encode(arg);
    view.set(buf);
    return {
        read: arg.length,
        written: buf.length
    };
});

function passStringToWasm0(arg, malloc, realloc) {

    if (realloc === undefined) {
        const buf = cachedTextEncoder.encode(arg);
        const ptr = malloc(buf.length, 1) >>> 0;
        getUint8ArrayMemory0().subarray(ptr, ptr + buf.length).set(buf);
        WASM_VECTOR_LEN = buf.length;
        return ptr;
    }

    let len = arg.length;
    let ptr = malloc(len, 1) >>> 0;

    const mem = getUint8ArrayMemory0();

    let offset = 0;

    for (; offset < len; offset++) {
        const code = arg.charCodeAt(offset);
        if (code > 0x7F) break;
        mem[ptr + offset] = code;
    }

    if (offset !== len) {
        if (offset !== 0) {
            arg = arg.slice(offset);
        }
        ptr = realloc(ptr, len, len = offset + arg.length * 3, 1) >>> 0;
        const view = getUint8ArrayMemory0().subarray(ptr + offset, ptr + len);
        const ret = encodeString(arg, view);

        offset += ret.written;
        ptr = realloc(ptr, len, offset, 1) >>> 0;
    }

    WASM_VECTOR_LEN = offset;
    return ptr;
}

let cachedDataViewMemory0 = null;

function getDataViewMemory0() {
    if (cachedDataViewMemory0 === null || cachedDataViewMemory0.buffer.detached === true || (cachedDataViewMemory0.buffer.detached === undefined && cachedDataViewMemory0.buffer !== wasm.memory.buffer)) {
        cachedDataViewMemory0 = new DataView(wasm.memory.buffer);
    }
    return cachedDataViewMemory0;
}

function addToExternrefTable0(obj) {
    const idx = wasm.__externref_table_alloc();
    wasm.__wbindgen_export_4.set(idx, obj);
    return idx;
}

function handleError(f, args) {
    try {
        return f.apply(this, args);
    } catch (e) {
        const idx = addToExternrefTable0(e);
        wasm.__wbindgen_exn_store(idx);
    }
}

const cachedTextDecoder = (typeof TextDecoder !== 'undefined' ? new TextDecoder('utf-8', { ignoreBOM: true, fatal: true }) : { decode: () => { throw Error('TextDecoder not available') } } );

if (typeof TextDecoder !== 'undefined') { cachedTextDecoder.decode(); };

function getStringFromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return cachedTextDecoder.decode(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}

function debugString(val) {
    // primitive types
    const type = typeof val;
    if (type == 'number' || type == 'boolean' || val == null) {
        return  `${val}`;
    }
    if (type == 'string') {
        return `"${val}"`;
    }
    if (type == 'symbol') {
        const description = val.description;
        if (description == null) {
            return 'Symbol';
        } else {
            return `Symbol(${description})`;
        }
    }
    if (type == 'function') {
        const name = val.name;
        if (typeof name == 'string' && name.length > 0) {
            return `Function(${name})`;
        } else {
            return 'Function';
        }
    }
    // objects
    if (Array.isArray(val)) {
        const length = val.length;
        let debug = '[';
        if (length > 0) {
            debug += debugString(val[0]);
        }
        for(let i = 1; i < length; i++) {
            debug += ', ' + debugString(val[i]);
        }
        debug += ']';
        return debug;
    }
    // Test for built-in
    const builtInMatches = /\[object ([^\]]+)\]/.exec(toString.call(val));
    let className;
    if (builtInMatches && builtInMatches.length > 1) {
        className = builtInMatches[1];
    } else {
        // Failed to match the standard '[object ClassName]'
        return toString.call(val);
    }
    if (className == 'Object') {
        // we're a user defined class or Object
        // JSON.stringify avoids problems with cycles, and is generally much
        // easier than looping through ownProperties of `val`.
        try {
            return 'Object(' + JSON.stringify(val) + ')';
        } catch (_) {
            return 'Object';
        }
    }
    // errors
    if (val instanceof Error) {
        return `${val.name}: ${val.message}\n${val.stack}`;
    }
    // TODO we could test for more things here, like `Set`s and `Map`s.
    return className;
}

function isLikeNone(x) {
    return x === undefined || x === null;
}

function takeFromExternrefTable0(idx) {
    const value = wasm.__wbindgen_export_4.get(idx);
    wasm.__externref_table_dealloc(idx);
    return value;
}

function _assertClass(instance, klass) {
    if (!(instance instanceof klass)) {
        throw new Error(`expected instance of ${klass.name}`);
    }
}
/**
 * Find commensurate angles for a given lattice
 * @param {WasmLattice2D} lattice
 * @param {number} max_index
 * @returns {any}
 */
export function find_commensurate_angles_wasm(lattice, max_index) {
    _assertClass(lattice, WasmLattice2D);
    const ret = wasm.find_commensurate_angles_wasm(lattice.__wbg_ptr, max_index);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
}

/**
 * Validate commensurability between two lattices
 * @param {WasmLattice2D} lattice_1
 * @param {WasmLattice2D} lattice_2
 * @param {number} tolerance
 * @returns {any}
 */
export function validate_commensurability_wasm(lattice_1, lattice_2, tolerance) {
    _assertClass(lattice_1, WasmLattice2D);
    _assertClass(lattice_2, WasmLattice2D);
    const ret = wasm.validate_commensurability_wasm(lattice_1.__wbg_ptr, lattice_2.__wbg_ptr, tolerance);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
}

let cachedFloat64ArrayMemory0 = null;

function getFloat64ArrayMemory0() {
    if (cachedFloat64ArrayMemory0 === null || cachedFloat64ArrayMemory0.byteLength === 0) {
        cachedFloat64ArrayMemory0 = new Float64Array(wasm.memory.buffer);
    }
    return cachedFloat64ArrayMemory0;
}

function getArrayF64FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getFloat64ArrayMemory0().subarray(ptr / 8, ptr / 8 + len);
}
/**
 * Compute moiré basis vectors from two lattices
 * @param {WasmLattice2D} lattice_1
 * @param {WasmLattice2D} lattice_2
 * @param {number} tolerance
 * @returns {Float64Array}
 */
export function compute_moire_basis_wasm(lattice_1, lattice_2, tolerance) {
    _assertClass(lattice_1, WasmLattice2D);
    _assertClass(lattice_2, WasmLattice2D);
    const ret = wasm.compute_moire_basis_wasm(lattice_1.__wbg_ptr, lattice_2.__wbg_ptr, tolerance);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v1;
}

function getArrayJsValueFromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    const mem = getDataViewMemory0();
    const result = [];
    for (let i = ptr; i < ptr + 4 * len; i += 4) {
        result.push(wasm.__wbindgen_export_4.get(mem.getUint32(i, true)));
    }
    wasm.__externref_drop_slice(ptr, len);
    return result;
}
/**
 * Analyze symmetries preserved in the moiré pattern
 * @param {WasmMoire2D} moire
 * @returns {string[]}
 */
export function analyze_moire_symmetry_wasm(moire) {
    _assertClass(moire, WasmMoire2D);
    const ret = wasm.analyze_moire_symmetry_wasm(moire.__wbg_ptr);
    var v1 = getArrayJsValueFromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
    return v1;
}

/**
 * Compute moiré potential at a given point
 * @param {WasmMoire2D} moire
 * @param {number} x
 * @param {number} y
 * @param {number} v_aa
 * @param {number} v_ab
 * @returns {number}
 */
export function moire_potential_at_wasm(moire, x, y, v_aa, v_ab) {
    _assertClass(moire, WasmMoire2D);
    const ret = wasm.moire_potential_at_wasm(moire.__wbg_ptr, x, y, v_aa, v_ab);
    return ret;
}

/**
 * Compute moiré potential over a grid for visualization
 * @param {WasmMoire2D} moire
 * @param {number} x_min
 * @param {number} x_max
 * @param {number} y_min
 * @param {number} y_max
 * @param {number} nx
 * @param {number} ny
 * @param {number} v_aa
 * @param {number} v_ab
 * @returns {any}
 */
export function compute_moire_potential_grid(moire, x_min, x_max, y_min, y_max, nx, ny, v_aa, v_ab) {
    _assertClass(moire, WasmMoire2D);
    const ret = wasm.compute_moire_potential_grid(moire.__wbg_ptr, x_min, x_max, y_min, y_max, nx, ny, v_aa, v_ab);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
}

/**
 * Find magic angles (special commensurate angles with interesting properties)
 * @param {WasmLattice2D} lattice
 * @returns {any}
 */
export function find_magic_angles(lattice) {
    _assertClass(lattice, WasmLattice2D);
    const ret = wasm.find_magic_angles(lattice.__wbg_ptr);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
}

/**
 * Analyze the quality of a moiré pattern based on commensurability and period
 * @param {WasmMoire2D} moire
 * @returns {any}
 */
export function analyze_moire_quality(moire) {
    _assertClass(moire, WasmMoire2D);
    const ret = wasm.analyze_moire_quality(moire.__wbg_ptr);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
}

/**
 * Get theoretical predictions for moiré properties
 * @param {number} lattice_constant
 * @param {number} twist_angle_degrees
 * @returns {any}
 */
export function get_moire_predictions(lattice_constant, twist_angle_degrees) {
    const ret = wasm.get_moire_predictions(lattice_constant, twist_angle_degrees);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
}

let cachedInt32ArrayMemory0 = null;

function getInt32ArrayMemory0() {
    if (cachedInt32ArrayMemory0 === null || cachedInt32ArrayMemory0.byteLength === 0) {
        cachedInt32ArrayMemory0 = new Int32Array(wasm.memory.buffer);
    }
    return cachedInt32ArrayMemory0;
}

function getArrayI32FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getInt32ArrayMemory0().subarray(ptr / 4, ptr / 4 + len);
}

function passArrayF64ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 8, 8) >>> 0;
    getFloat64ArrayMemory0().set(arg, ptr / 8);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}
/**
 * Create a simple twisted bilayer moiré pattern
 * @param {WasmLattice2D} lattice
 * @param {number} angle_degrees
 * @returns {WasmMoire2D}
 */
export function create_twisted_bilayer(lattice, angle_degrees) {
    _assertClass(lattice, WasmLattice2D);
    const ret = wasm.create_twisted_bilayer(lattice.__wbg_ptr, angle_degrees);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return WasmMoire2D.__wrap(ret[0]);
}

/**
 * Create a moiré pattern with commensurate angle
 * @param {WasmLattice2D} lattice
 * @param {number} m1
 * @param {number} m2
 * @param {number} n1
 * @param {number} n2
 * @returns {WasmMoire2D}
 */
export function create_commensurate_moire(lattice, m1, m2, n1, n2) {
    _assertClass(lattice, WasmLattice2D);
    const ret = wasm.create_commensurate_moire(lattice.__wbg_ptr, m1, m2, n1, n2);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return WasmMoire2D.__wrap(ret[0]);
}

/**
 * Create twisted bilayer graphene moiré pattern with magic angle
 * @param {WasmLattice2D} lattice
 * @returns {WasmMoire2D}
 */
export function create_magic_angle_graphene(lattice) {
    _assertClass(lattice, WasmLattice2D);
    const ret = wasm.create_magic_angle_graphene(lattice.__wbg_ptr);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return WasmMoire2D.__wrap(ret[0]);
}

/**
 * Create a series of moiré patterns with different twist angles
 * @param {WasmLattice2D} lattice
 * @param {number} start_angle
 * @param {number} end_angle
 * @param {number} num_steps
 * @returns {WasmMoire2D[]}
 */
export function create_twist_series(lattice, start_angle, end_angle, num_steps) {
    _assertClass(lattice, WasmLattice2D);
    const ret = wasm.create_twist_series(lattice.__wbg_ptr, start_angle, end_angle, num_steps);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v1 = getArrayJsValueFromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
    return v1;
}

/**
 * Get recommended twist angles for studying moiré patterns
 * @returns {Float64Array}
 */
export function get_recommended_twist_angles() {
    const ret = wasm.get_recommended_twist_angles();
    var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v1;
}

/**
 * Calculate the expected moiré period for a given twist angle and lattice constant
 * @param {number} lattice_constant
 * @param {number} twist_angle_degrees
 * @returns {number}
 */
export function calculate_moire_period(lattice_constant, twist_angle_degrees) {
    const ret = wasm.calculate_moire_period(lattice_constant, twist_angle_degrees);
    return ret;
}

/**
 * Get transformation matrix for rotation and scaling
 * @param {number} angle_degrees
 * @param {number} scale
 * @returns {Float64Array}
 */
export function get_rotation_scale_matrix(angle_degrees, scale) {
    const ret = wasm.get_rotation_scale_matrix(angle_degrees, scale);
    var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v1;
}

/**
 * Get transformation matrix for anisotropic scaling
 * @param {number} scale_x
 * @param {number} scale_y
 * @returns {Float64Array}
 */
export function get_anisotropic_scale_matrix(scale_x, scale_y) {
    const ret = wasm.get_anisotropic_scale_matrix(scale_x, scale_y);
    var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v1;
}

/**
 * Get transformation matrix for shear
 * @param {number} shear_x
 * @param {number} shear_y
 * @returns {Float64Array}
 */
export function get_shear_matrix(shear_x, shear_y) {
    const ret = wasm.get_shear_matrix(shear_x, shear_y);
    var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v1;
}

/**
 * Compute Wigner-Seitz cell for 2D lattice
 * @param {Float64Array} basis
 * @param {number} tolerance
 * @returns {WasmPolyhedron}
 */
export function compute_wigner_seitz_2d(basis, tolerance) {
    const ptr0 = passArrayF64ToWasm0(basis, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.compute_wigner_seitz_2d(ptr0, len0, tolerance);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return WasmPolyhedron.__wrap(ret[0]);
}

/**
 * Compute Wigner-Seitz cell for 3D lattice
 * @param {Float64Array} basis
 * @param {number} tolerance
 * @returns {WasmPolyhedron}
 */
export function compute_wigner_seitz_3d(basis, tolerance) {
    const ptr0 = passArrayF64ToWasm0(basis, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.compute_wigner_seitz_3d(ptr0, len0, tolerance);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return WasmPolyhedron.__wrap(ret[0]);
}

/**
 * Compute Brillouin zone for 2D lattice
 * @param {Float64Array} reciprocal_basis
 * @param {number} tolerance
 * @returns {WasmPolyhedron}
 */
export function compute_brillouin_2d(reciprocal_basis, tolerance) {
    const ptr0 = passArrayF64ToWasm0(reciprocal_basis, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.compute_brillouin_2d(ptr0, len0, tolerance);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return WasmPolyhedron.__wrap(ret[0]);
}

/**
 * Compute Brillouin zone for 3D lattice
 * @param {Float64Array} reciprocal_basis
 * @param {number} tolerance
 * @returns {WasmPolyhedron}
 */
export function compute_brillouin_3d(reciprocal_basis, tolerance) {
    const ptr0 = passArrayF64ToWasm0(reciprocal_basis, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.compute_brillouin_3d(ptr0, len0, tolerance);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return WasmPolyhedron.__wrap(ret[0]);
}

export function main() {
    wasm.main();
}

/**
 * Get the version of the library
 * @returns {string}
 */
export function version() {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.version();
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
    }
}

/**
 * Identify Bravais lattice type for 2D from metric tensor
 * @param {Float64Array} metric
 * @param {number} tolerance
 * @returns {WasmBravais2D}
 */
export function identify_bravais_type_2d(metric, tolerance) {
    const ptr0 = passArrayF64ToWasm0(metric, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.identify_bravais_type_2d(ptr0, len0, tolerance);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0];
}

/**
 * Identify Bravais lattice type for 3D from metric tensor
 * @param {Float64Array} metric
 * @param {number} tolerance
 * @returns {WasmBravais3D}
 */
export function identify_bravais_type_3d(metric, tolerance) {
    const ptr0 = passArrayF64ToWasm0(metric, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.identify_bravais_type_3d(ptr0, len0, tolerance);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0];
}

/**
 * Create square lattice
 * @param {number} a
 * @returns {WasmLattice2D}
 */
export function create_square_lattice(a) {
    const ret = wasm.create_square_lattice(a);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return WasmLattice2D.__wrap(ret[0]);
}

/**
 * Create hexagonal lattice
 * @param {number} a
 * @returns {WasmLattice2D}
 */
export function create_hexagonal_lattice(a) {
    const ret = wasm.create_hexagonal_lattice(a);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return WasmLattice2D.__wrap(ret[0]);
}

/**
 * Create rectangular lattice
 * @param {number} a
 * @param {number} b
 * @returns {WasmLattice2D}
 */
export function create_rectangular_lattice(a, b) {
    const ret = wasm.create_rectangular_lattice(a, b);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return WasmLattice2D.__wrap(ret[0]);
}

/**
 * Create centered rectangular lattice
 * @param {number} a
 * @param {number} b
 * @returns {WasmLattice2D}
 */
export function create_centered_rectangular_lattice(a, b) {
    const ret = wasm.create_centered_rectangular_lattice(a, b);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return WasmLattice2D.__wrap(ret[0]);
}

/**
 * Create oblique lattice
 * @param {number} a
 * @param {number} b
 * @param {number} gamma_degrees
 * @returns {WasmLattice2D}
 */
export function create_oblique_lattice(a, b, gamma_degrees) {
    const ret = wasm.create_oblique_lattice(a, b, gamma_degrees);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return WasmLattice2D.__wrap(ret[0]);
}

/**
 * Create body-centered cubic lattice
 * @param {number} a
 * @returns {WasmLattice3D}
 */
export function create_body_centered_cubic_lattice(a) {
    const ret = wasm.create_body_centered_cubic_lattice(a);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return WasmLattice3D.__wrap(ret[0]);
}

/**
 * Create face-centered cubic lattice
 * @param {number} a
 * @returns {WasmLattice3D}
 */
export function create_face_centered_cubic_lattice(a) {
    const ret = wasm.create_face_centered_cubic_lattice(a);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return WasmLattice3D.__wrap(ret[0]);
}

/**
 * Create hexagonal close-packed lattice
 * @param {number} a
 * @param {number} c
 * @returns {WasmLattice3D}
 */
export function create_hexagonal_close_packed_lattice(a, c) {
    const ret = wasm.create_hexagonal_close_packed_lattice(a, c);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return WasmLattice3D.__wrap(ret[0]);
}

/**
 * Create tetragonal lattice
 * @param {number} a
 * @param {number} c
 * @returns {WasmLattice3D}
 */
export function create_tetragonal_lattice(a, c) {
    const ret = wasm.create_tetragonal_lattice(a, c);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return WasmLattice3D.__wrap(ret[0]);
}

/**
 * Create orthorhombic lattice
 * @param {number} a
 * @param {number} b
 * @param {number} c
 * @returns {WasmLattice3D}
 */
export function create_orthorhombic_lattice(a, b, c) {
    const ret = wasm.create_orthorhombic_lattice(a, b, c);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return WasmLattice3D.__wrap(ret[0]);
}

/**
 * Create rhombohedral lattice
 * @param {number} a
 * @param {number} alpha_degrees
 * @returns {WasmLattice3D}
 */
export function create_rhombohedral_lattice(a, alpha_degrees) {
    const ret = wasm.create_rhombohedral_lattice(a, alpha_degrees);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return WasmLattice3D.__wrap(ret[0]);
}

/**
 * Scale 2D lattice uniformly
 * @param {WasmLattice2D} lattice
 * @param {number} scale
 * @returns {WasmLattice2D}
 */
export function scale_2d_lattice(lattice, scale) {
    _assertClass(lattice, WasmLattice2D);
    const ret = wasm.scale_2d_lattice(lattice.__wbg_ptr, scale);
    return WasmLattice2D.__wrap(ret);
}

/**
 * Scale 3D lattice uniformly
 * @param {WasmLattice3D} lattice
 * @param {number} scale
 * @returns {WasmLattice3D}
 */
export function scale_3d_lattice(lattice, scale) {
    _assertClass(lattice, WasmLattice3D);
    const ret = wasm.scale_3d_lattice(lattice.__wbg_ptr, scale);
    return WasmLattice3D.__wrap(ret);
}

/**
 * Rotate 2D lattice by angle in degrees
 * @param {WasmLattice2D} lattice
 * @param {number} angle_degrees
 * @returns {WasmLattice2D}
 */
export function rotate_2d_lattice(lattice, angle_degrees) {
    _assertClass(lattice, WasmLattice2D);
    const ret = wasm.rotate_2d_lattice(lattice.__wbg_ptr, angle_degrees);
    return WasmLattice2D.__wrap(ret);
}

/**
 * Create 2D supercell
 * @param {WasmLattice2D} lattice
 * @param {number} nx
 * @param {number} ny
 * @returns {WasmLattice2D}
 */
export function create_2d_supercell(lattice, nx, ny) {
    _assertClass(lattice, WasmLattice2D);
    const ret = wasm.create_2d_supercell(lattice.__wbg_ptr, nx, ny);
    return WasmLattice2D.__wrap(ret);
}

/**
 * Create 3D supercell
 * @param {WasmLattice3D} lattice
 * @param {number} nx
 * @param {number} ny
 * @param {number} nz
 * @returns {WasmLattice3D}
 */
export function create_3d_supercell(lattice, nx, ny, nz) {
    _assertClass(lattice, WasmLattice3D);
    const ret = wasm.create_3d_supercell(lattice.__wbg_ptr, nx, ny, nz);
    return WasmLattice3D.__wrap(ret);
}

/**
 * WASM wrapper for Bravais2D enum
 * @enum {0 | 1 | 2 | 3 | 4}
 */
export const WasmBravais2D = Object.freeze({
    Square: 0, "0": "Square",
    Hexagonal: 1, "1": "Hexagonal",
    Rectangular: 2, "2": "Rectangular",
    CenteredRectangular: 3, "3": "CenteredRectangular",
    Oblique: 4, "4": "Oblique",
});
/**
 * WASM wrapper for Bravais3D enum
 * @enum {0 | 1 | 2 | 3 | 4 | 5 | 6}
 */
export const WasmBravais3D = Object.freeze({
    Cubic: 0, "0": "Cubic",
    Tetragonal: 1, "1": "Tetragonal",
    Orthorhombic: 2, "2": "Orthorhombic",
    Hexagonal: 3, "3": "Hexagonal",
    Trigonal: 4, "4": "Trigonal",
    Monoclinic: 5, "5": "Monoclinic",
    Triclinic: 6, "6": "Triclinic",
});
/**
 * WASM wrapper for Centering enum
 * @enum {0 | 1 | 2 | 3}
 */
export const WasmCentering = Object.freeze({
    Primitive: 0, "0": "Primitive",
    BodyCentered: 1, "1": "BodyCentered",
    FaceCentered: 2, "2": "FaceCentered",
    BaseCentered: 3, "3": "BaseCentered",
});
/**
 * WASM wrapper for MoireTransformation enum
 * @enum {0 | 1 | 2 | 3}
 */
export const WasmMoireTransformation = Object.freeze({
    RotationScale: 0, "0": "RotationScale",
    AnisotropicScale: 1, "1": "AnisotropicScale",
    Shear: 2, "2": "Shear",
    General: 3, "3": "General",
});

const WasmLattice2DFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmlattice2d_free(ptr >>> 0, 1));
/**
 * WASM wrapper for 2D lattice
 */
export class WasmLattice2D {

    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(WasmLattice2D.prototype);
        obj.__wbg_ptr = ptr;
        WasmLattice2DFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmLattice2DFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmlattice2d_free(ptr, 0);
    }
    /**
     * Create a new lattice from JavaScript parameters
     * @param {any} params
     */
    constructor(params) {
        const ret = wasm.wasmlattice2d_new(params);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0] >>> 0;
        WasmLattice2DFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Generate lattice points within a radius
     * @param {number} radius
     * @param {number} center_x
     * @param {number} center_y
     * @returns {any}
     */
    generate_points(radius, center_x, center_y) {
        const ret = wasm.wasmlattice2d_generate_points(this.__wbg_ptr, radius, center_x, center_y);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * Get lattice parameters as JavaScript object
     * @returns {any}
     */
    get_parameters() {
        const ret = wasm.wasmlattice2d_get_parameters(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * Get unit cell area
     * @returns {number}
     */
    unit_cell_area() {
        const ret = wasm.wasmlattice2d_unit_cell_area(this.__wbg_ptr);
        return ret;
    }
    /**
     * Get lattice vectors as JavaScript object
     * @returns {any}
     */
    lattice_vectors() {
        const ret = wasm.wasmlattice2d_lattice_vectors(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * Get reciprocal lattice vectors
     * @returns {any}
     */
    reciprocal_vectors() {
        const ret = wasm.wasmlattice2d_reciprocal_vectors(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * Generate an SVG representation of the lattice
     * @param {number} width
     * @param {number} height
     * @param {number} radius
     * @returns {string}
     */
    to_svg(width, height, radius) {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.wasmlattice2d_to_svg(this.__wbg_ptr, width, height, radius);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Convert fractional to cartesian coordinates
     * @param {number} fx
     * @param {number} fy
     * @returns {any}
     */
    frac_to_cart(fx, fy) {
        const ret = wasm.wasmlattice2d_frac_to_cart(this.__wbg_ptr, fx, fy);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * Convert cartesian to fractional coordinates
     * @param {number} x
     * @param {number} y
     * @returns {any}
     */
    cart_to_frac(x, y) {
        const ret = wasm.wasmlattice2d_cart_to_frac(this.__wbg_ptr, x, y);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * Get Bravais lattice type
     * @returns {WasmBravais2D}
     */
    bravais_type() {
        const ret = wasm.wasmlattice2d_bravais_type(this.__wbg_ptr);
        return ret;
    }
    /**
     * Check if k-point is in Brillouin zone
     * @param {number} kx
     * @param {number} ky
     * @returns {boolean}
     */
    in_brillouin_zone(kx, ky) {
        const ret = wasm.wasmlattice2d_in_brillouin_zone(this.__wbg_ptr, kx, ky);
        return ret !== 0;
    }
    /**
     * Reduce k-point to first Brillouin zone
     * @param {number} kx
     * @param {number} ky
     * @returns {any}
     */
    reduce_to_brillouin_zone(kx, ky) {
        const ret = wasm.wasmlattice2d_reduce_to_brillouin_zone(this.__wbg_ptr, kx, ky);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * Get Wigner-Seitz cell
     * @returns {WasmPolyhedron}
     */
    wigner_seitz_cell() {
        const ret = wasm.wasmlattice2d_wigner_seitz_cell(this.__wbg_ptr);
        return WasmPolyhedron.__wrap(ret);
    }
    /**
     * Get Brillouin zone
     * @returns {WasmPolyhedron}
     */
    brillouin_zone() {
        const ret = wasm.wasmlattice2d_brillouin_zone(this.__wbg_ptr);
        return WasmPolyhedron.__wrap(ret);
    }
    /**
     * Get coordination analysis
     * @returns {any}
     */
    coordination_analysis() {
        const ret = wasm.wasmlattice2d_coordination_analysis(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * Get packing fraction for given atomic radius
     * @param {number} _radius
     * @returns {number}
     */
    packing_fraction(_radius) {
        const ret = wasm.wasmlattice2d_packing_fraction(this.__wbg_ptr, _radius);
        return ret;
    }
    /**
     * Extend to 3D lattice with given c-vector
     * @param {number} cx
     * @param {number} cy
     * @param {number} cz
     * @returns {WasmLattice3D}
     */
    to_3d(cx, cy, cz) {
        const ret = wasm.wasmlattice2d_to_3d(this.__wbg_ptr, cx, cy, cz);
        return WasmLattice3D.__wrap(ret);
    }
    /**
     * Generate lattice points by shell
     * @param {number} max_shell
     * @returns {any}
     */
    generate_points_by_shell(max_shell) {
        const ret = wasm.wasmlattice2d_generate_points_by_shell(this.__wbg_ptr, max_shell);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * Generate direct-space lattice points in a rectangle
     * @param {number} width
     * @param {number} height
     * @returns {any}
     */
    get_direct_lattice_points_in_rectangle(width, height) {
        const ret = wasm.wasmlattice2d_get_direct_lattice_points_in_rectangle(this.__wbg_ptr, width, height);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * Generate reciprocal-space lattice points in a rectangle
     * @param {number} width
     * @param {number} height
     * @returns {any}
     */
    get_reciprocal_lattice_points_in_rectangle(width, height) {
        const ret = wasm.wasmlattice2d_get_reciprocal_lattice_points_in_rectangle(this.__wbg_ptr, width, height);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * Get high symmetry points in Cartesian coordinates
     * @returns {any}
     */
    get_high_symmetry_points() {
        const ret = wasm.wasmlattice2d_get_high_symmetry_points(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * Get high symmetry path data
     * @returns {any}
     */
    get_high_symmetry_path() {
        const ret = wasm.wasmlattice2d_get_high_symmetry_path(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
}

const WasmLattice3DFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmlattice3d_free(ptr >>> 0, 1));
/**
 * WASM wrapper for 3D lattice
 */
export class WasmLattice3D {

    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(WasmLattice3D.prototype);
        obj.__wbg_ptr = ptr;
        WasmLattice3DFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmLattice3DFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmlattice3d_free(ptr, 0);
    }
    /**
     * Create a new 3D lattice from JavaScript parameters
     * @param {any} params
     */
    constructor(params) {
        const ret = wasm.wasmlattice3d_new(params);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0] >>> 0;
        WasmLattice3DFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Convert fractional to cartesian coordinates
     * @param {number} fx
     * @param {number} fy
     * @param {number} fz
     * @returns {any}
     */
    frac_to_cart(fx, fy, fz) {
        const ret = wasm.wasmlattice3d_frac_to_cart(this.__wbg_ptr, fx, fy, fz);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * Convert cartesian to fractional coordinates
     * @param {number} x
     * @param {number} y
     * @param {number} z
     * @returns {any}
     */
    cart_to_frac(x, y, z) {
        const ret = wasm.wasmlattice3d_cart_to_frac(this.__wbg_ptr, x, y, z);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * Get lattice parameters
     * @returns {any}
     */
    lattice_parameters() {
        const ret = wasm.wasmlattice3d_lattice_parameters(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * Get lattice angles in degrees
     * @returns {any}
     */
    lattice_angles() {
        const ret = wasm.wasmlattice3d_lattice_angles(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * Get cell volume
     * @returns {number}
     */
    cell_volume() {
        const ret = wasm.wasmlattice3d_cell_volume(this.__wbg_ptr);
        return ret;
    }
    /**
     * Get Bravais lattice type
     * @returns {WasmBravais3D}
     */
    bravais_type() {
        const ret = wasm.wasmlattice3d_bravais_type(this.__wbg_ptr);
        return ret;
    }
    /**
     * Check if k-point is in Brillouin zone
     * @param {number} kx
     * @param {number} ky
     * @param {number} kz
     * @returns {boolean}
     */
    in_brillouin_zone(kx, ky, kz) {
        const ret = wasm.wasmlattice3d_in_brillouin_zone(this.__wbg_ptr, kx, ky, kz);
        return ret !== 0;
    }
    /**
     * Reduce k-point to first Brillouin zone
     * @param {number} kx
     * @param {number} ky
     * @param {number} kz
     * @returns {any}
     */
    reduce_to_brillouin_zone(kx, ky, kz) {
        const ret = wasm.wasmlattice3d_reduce_to_brillouin_zone(this.__wbg_ptr, kx, ky, kz);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * Generate 3D lattice points within radius
     * @param {number} radius
     * @returns {any}
     */
    generate_points_3d(radius) {
        const ret = wasm.wasmlattice3d_generate_points_3d(this.__wbg_ptr, radius);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * Generate 3D lattice points by shell
     * @param {number} max_shell
     * @returns {any}
     */
    generate_points_3d_by_shell(max_shell) {
        const ret = wasm.wasmlattice3d_generate_points_3d_by_shell(this.__wbg_ptr, max_shell);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * Get Wigner-Seitz cell
     * @returns {WasmPolyhedron}
     */
    wigner_seitz_cell() {
        const ret = wasm.wasmlattice3d_wigner_seitz_cell(this.__wbg_ptr);
        return WasmPolyhedron.__wrap(ret);
    }
    /**
     * Get Brillouin zone
     * @returns {WasmPolyhedron}
     */
    brillouin_zone() {
        const ret = wasm.wasmlattice3d_brillouin_zone(this.__wbg_ptr);
        return WasmPolyhedron.__wrap(ret);
    }
    /**
     * Get coordination analysis
     * @returns {any}
     */
    coordination_analysis() {
        const ret = wasm.wasmlattice3d_coordination_analysis(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * Get packing fraction for given atomic radius
     * @param {number} _radius
     * @returns {number}
     */
    packing_fraction(_radius) {
        const ret = wasm.wasmlattice3d_packing_fraction(this.__wbg_ptr, _radius);
        return ret;
    }
    /**
     * Convert to 2D lattice (projection onto a-b plane)
     * @returns {WasmLattice2D}
     */
    to_2d() {
        const ret = wasm.wasmlattice3d_to_2d(this.__wbg_ptr);
        return WasmLattice2D.__wrap(ret);
    }
    /**
     * Get high symmetry points in Cartesian coordinates
     * @returns {any}
     */
    get_high_symmetry_points() {
        const ret = wasm.wasmlattice3d_get_high_symmetry_points(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * Get high symmetry path data
     * @returns {any}
     */
    get_high_symmetry_path() {
        const ret = wasm.wasmlattice3d_get_high_symmetry_path(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
}

const WasmMoire2DFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmmoire2d_free(ptr >>> 0, 1));
/**
 * WASM wrapper for 2D moiré lattice
 */
export class WasmMoire2D {

    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(WasmMoire2D.prototype);
        obj.__wbg_ptr = ptr;
        WasmMoire2DFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmMoire2DFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmmoire2d_free(ptr, 0);
    }
    /**
     * Get the moiré lattice as a regular 2D lattice
     * @returns {WasmLattice2D}
     */
    as_lattice2d() {
        const ret = wasm.wasmmoire2d_as_lattice2d(this.__wbg_ptr);
        return WasmLattice2D.__wrap(ret);
    }
    /**
     * Get moiré primitive vectors as JavaScript object
     * @returns {any}
     */
    primitive_vectors() {
        const ret = wasm.wasmmoire2d_primitive_vectors(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * Get the moiré periodicity ratio
     * @returns {number}
     */
    moire_period_ratio() {
        const ret = wasm.wasmmoire2d_moire_period_ratio(this.__wbg_ptr);
        return ret;
    }
    /**
     * Check if a point belongs to lattice 1
     * @param {number} x
     * @param {number} y
     * @returns {boolean}
     */
    is_lattice1_point(x, y) {
        const ret = wasm.wasmmoire2d_is_lattice1_point(this.__wbg_ptr, x, y);
        return ret !== 0;
    }
    /**
     * Check if a point belongs to lattice 2
     * @param {number} x
     * @param {number} y
     * @returns {boolean}
     */
    is_lattice2_point(x, y) {
        const ret = wasm.wasmmoire2d_is_lattice2_point(this.__wbg_ptr, x, y);
        return ret !== 0;
    }
    /**
     * Get stacking type at a given position
     * @param {number} x
     * @param {number} y
     * @returns {string | undefined}
     */
    get_stacking_at(x, y) {
        const ret = wasm.wasmmoire2d_get_stacking_at(this.__wbg_ptr, x, y);
        let v1;
        if (ret[0] !== 0) {
            v1 = getStringFromWasm0(ret[0], ret[1]).slice();
            wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
        }
        return v1;
    }
    /**
     * Get the twist angle in degrees
     * @returns {number}
     */
    twist_angle_degrees() {
        const ret = wasm.wasmmoire2d_twist_angle_degrees(this.__wbg_ptr);
        return ret;
    }
    /**
     * Get the twist angle in radians
     * @returns {number}
     */
    twist_angle_radians() {
        const ret = wasm.wasmmoire2d_twist_angle_radians(this.__wbg_ptr);
        return ret;
    }
    /**
     * Check if the moiré lattice is commensurate
     * @returns {boolean}
     */
    is_commensurate() {
        const ret = wasm.wasmmoire2d_is_commensurate(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * Get coincidence indices if commensurate
     * @returns {Int32Array | undefined}
     */
    coincidence_indices() {
        const ret = wasm.wasmmoire2d_coincidence_indices(this.__wbg_ptr);
        let v1;
        if (ret[0] !== 0) {
            v1 = getArrayI32FromWasm0(ret[0], ret[1]).slice();
            wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        }
        return v1;
    }
    /**
     * Get the first constituent lattice
     * @returns {WasmLattice2D}
     */
    lattice_1() {
        const ret = wasm.wasmmoire2d_lattice_1(this.__wbg_ptr);
        return WasmLattice2D.__wrap(ret);
    }
    /**
     * Get the second constituent lattice
     * @returns {WasmLattice2D}
     */
    lattice_2() {
        const ret = wasm.wasmmoire2d_lattice_2(this.__wbg_ptr);
        return WasmLattice2D.__wrap(ret);
    }
    /**
     * Get unit cell area of the moiré lattice
     * @returns {number}
     */
    cell_area() {
        const ret = wasm.wasmmoire2d_cell_area(this.__wbg_ptr);
        return ret;
    }
    /**
     * Get transformation matrix as JavaScript array (flattened 2x2 matrix)
     * @returns {Float64Array}
     */
    transformation_matrix() {
        const ret = wasm.wasmmoire2d_transformation_matrix(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * Get lattice parameters as JavaScript object
     * @returns {any}
     */
    get_parameters() {
        const ret = wasm.wasmmoire2d_get_parameters(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * Generate lattice points within a radius for visualization
     * @param {number} radius
     * @returns {any}
     */
    generate_moire_points(radius) {
        const ret = wasm.wasmmoire2d_generate_moire_points(this.__wbg_ptr, radius);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * Generate lattice 1 points within a radius
     * @param {number} radius
     * @returns {any}
     */
    generate_lattice1_points(radius) {
        const ret = wasm.wasmmoire2d_generate_lattice1_points(this.__wbg_ptr, radius);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * Generate lattice 2 points within a radius
     * @param {number} radius
     * @returns {any}
     */
    generate_lattice2_points(radius) {
        const ret = wasm.wasmmoire2d_generate_lattice2_points(this.__wbg_ptr, radius);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * Get stacking analysis for points within a radius
     * @param {number} radius
     * @param {number} grid_spacing
     * @returns {any}
     */
    analyze_stacking_in_region(radius, grid_spacing) {
        const ret = wasm.wasmmoire2d_analyze_stacking_in_region(this.__wbg_ptr, radius, grid_spacing);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * Convert fractional to cartesian coordinates using moiré basis
     * @param {number} fx
     * @param {number} fy
     * @returns {any}
     */
    frac_to_cart(fx, fy) {
        const ret = wasm.wasmmoire2d_frac_to_cart(this.__wbg_ptr, fx, fy);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * Convert cartesian to fractional coordinates using moiré basis
     * @param {number} x
     * @param {number} y
     * @returns {any}
     */
    cart_to_frac(x, y) {
        const ret = wasm.wasmmoire2d_cart_to_frac(this.__wbg_ptr, x, y);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
}

const WasmMoireBuilderFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmmoirebuilder_free(ptr >>> 0, 1));
/**
 * WASM wrapper for MoireBuilder
 */
export class WasmMoireBuilder {

    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(WasmMoireBuilder.prototype);
        obj.__wbg_ptr = ptr;
        WasmMoireBuilderFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmMoireBuilderFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmmoirebuilder_free(ptr, 0);
    }
    /**
     * Create a new MoireBuilder
     */
    constructor() {
        const ret = wasm.wasmmoirebuilder_new();
        this.__wbg_ptr = ret >>> 0;
        WasmMoireBuilderFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Set the base lattice
     * @param {WasmLattice2D} lattice
     * @returns {WasmMoireBuilder}
     */
    with_base_lattice(lattice) {
        const ptr = this.__destroy_into_raw();
        _assertClass(lattice, WasmLattice2D);
        const ret = wasm.wasmmoirebuilder_with_base_lattice(ptr, lattice.__wbg_ptr);
        return WasmMoireBuilder.__wrap(ret);
    }
    /**
     * Set tolerance for calculations
     * @param {number} tolerance
     * @returns {WasmMoireBuilder}
     */
    with_tolerance(tolerance) {
        const ptr = this.__destroy_into_raw();
        const ret = wasm.wasmmoirebuilder_with_tolerance(ptr, tolerance);
        return WasmMoireBuilder.__wrap(ret);
    }
    /**
     * Set a rotation and uniform scaling transformation
     * @param {number} angle_degrees
     * @param {number} scale
     * @returns {WasmMoireBuilder}
     */
    with_twist_and_scale(angle_degrees, scale) {
        const ptr = this.__destroy_into_raw();
        const ret = wasm.wasmmoirebuilder_with_twist_and_scale(ptr, angle_degrees, scale);
        return WasmMoireBuilder.__wrap(ret);
    }
    /**
     * Set an anisotropic scaling transformation
     * @param {number} scale_x
     * @param {number} scale_y
     * @returns {WasmMoireBuilder}
     */
    with_anisotropic_scale(scale_x, scale_y) {
        const ptr = this.__destroy_into_raw();
        const ret = wasm.wasmmoirebuilder_with_anisotropic_scale(ptr, scale_x, scale_y);
        return WasmMoireBuilder.__wrap(ret);
    }
    /**
     * Set a shear transformation
     * @param {number} shear_x
     * @param {number} shear_y
     * @returns {WasmMoireBuilder}
     */
    with_shear(shear_x, shear_y) {
        const ptr = this.__destroy_into_raw();
        const ret = wasm.wasmmoirebuilder_with_shear(ptr, shear_x, shear_y);
        return WasmMoireBuilder.__wrap(ret);
    }
    /**
     * Set a general 2x2 transformation matrix (flattened array)
     * @param {Float64Array} matrix
     * @returns {WasmMoireBuilder}
     */
    with_general_transformation(matrix) {
        const ptr = this.__destroy_into_raw();
        const ptr0 = passArrayF64ToWasm0(matrix, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmmoirebuilder_with_general_transformation(ptr, ptr0, len0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return WasmMoireBuilder.__wrap(ret[0]);
    }
    /**
     * Build the Moire2D lattice
     * @returns {WasmMoire2D}
     */
    build() {
        const ptr = this.__destroy_into_raw();
        const ret = wasm.wasmmoirebuilder_build(ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return WasmMoire2D.__wrap(ret[0]);
    }
    /**
     * Build with JavaScript parameters object
     * @param {WasmLattice2D} lattice
     * @param {any} params
     * @returns {WasmMoire2D}
     */
    static build_with_params(lattice, params) {
        _assertClass(lattice, WasmLattice2D);
        const ret = wasm.wasmmoirebuilder_build_with_params(lattice.__wbg_ptr, params);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return WasmMoire2D.__wrap(ret[0]);
    }
}

const WasmPolyhedronFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmpolyhedron_free(ptr >>> 0, 1));
/**
 * WASM wrapper for Polyhedron
 */
export class WasmPolyhedron {

    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(WasmPolyhedron.prototype);
        obj.__wbg_ptr = ptr;
        WasmPolyhedronFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmPolyhedronFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmpolyhedron_free(ptr, 0);
    }
    /**
     * Check if a 2D point is inside the polyhedron
     * @param {number} x
     * @param {number} y
     * @returns {boolean}
     */
    contains_2d(x, y) {
        const ret = wasm.wasmpolyhedron_contains_2d(this.__wbg_ptr, x, y);
        return ret !== 0;
    }
    /**
     * Check if a 3D point is inside the polyhedron
     * @param {number} x
     * @param {number} y
     * @param {number} z
     * @returns {boolean}
     */
    contains_3d(x, y, z) {
        const ret = wasm.wasmpolyhedron_contains_3d(this.__wbg_ptr, x, y, z);
        return ret !== 0;
    }
    /**
     * Get the measure (area for 2D, volume for 3D)
     * @returns {number}
     */
    measure() {
        const ret = wasm.wasmpolyhedron_measure(this.__wbg_ptr);
        return ret;
    }
    /**
     * Get polyhedron data as JavaScript object
     * @returns {any}
     */
    get_data() {
        const ret = wasm.wasmpolyhedron_get_data(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
}

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);

            } catch (e) {
                if (module.headers.get('Content-Type') != 'application/wasm') {
                    console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);

                } else {
                    throw e;
                }
            }
        }

        const bytes = await module.arrayBuffer();
        return await WebAssembly.instantiate(bytes, imports);

    } else {
        const instance = await WebAssembly.instantiate(module, imports);

        if (instance instanceof WebAssembly.Instance) {
            return { instance, module };

        } else {
            return instance;
        }
    }
}

function __wbg_get_imports() {
    const imports = {};
    imports.wbg = {};
    imports.wbg.__wbg_String_8f0eb39a4a4c2f66 = function(arg0, arg1) {
        const ret = String(arg1);
        const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
        getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
    };
    imports.wbg.__wbg_buffer_609cc3eee51ed158 = function(arg0) {
        const ret = arg0.buffer;
        return ret;
    };
    imports.wbg.__wbg_call_672a4d21634d4a24 = function() { return handleError(function (arg0, arg1) {
        const ret = arg0.call(arg1);
        return ret;
    }, arguments) };
    imports.wbg.__wbg_done_769e5ede4b31c67b = function(arg0) {
        const ret = arg0.done;
        return ret;
    };
    imports.wbg.__wbg_error_7534b8e9a36f1ab4 = function(arg0, arg1) {
        let deferred0_0;
        let deferred0_1;
        try {
            deferred0_0 = arg0;
            deferred0_1 = arg1;
            console.error(getStringFromWasm0(arg0, arg1));
        } finally {
            wasm.__wbindgen_free(deferred0_0, deferred0_1, 1);
        }
    };
    imports.wbg.__wbg_get_67b2ba62fc30de12 = function() { return handleError(function (arg0, arg1) {
        const ret = Reflect.get(arg0, arg1);
        return ret;
    }, arguments) };
    imports.wbg.__wbg_get_b9b93047fe3cf45b = function(arg0, arg1) {
        const ret = arg0[arg1 >>> 0];
        return ret;
    };
    imports.wbg.__wbg_getwithrefkey_1dc361bd10053bfe = function(arg0, arg1) {
        const ret = arg0[arg1];
        return ret;
    };
    imports.wbg.__wbg_instanceof_ArrayBuffer_e14585432e3737fc = function(arg0) {
        let result;
        try {
            result = arg0 instanceof ArrayBuffer;
        } catch (_) {
            result = false;
        }
        const ret = result;
        return ret;
    };
    imports.wbg.__wbg_instanceof_Uint8Array_17156bcf118086a9 = function(arg0) {
        let result;
        try {
            result = arg0 instanceof Uint8Array;
        } catch (_) {
            result = false;
        }
        const ret = result;
        return ret;
    };
    imports.wbg.__wbg_isArray_a1eab7e0d067391b = function(arg0) {
        const ret = Array.isArray(arg0);
        return ret;
    };
    imports.wbg.__wbg_iterator_9a24c88df860dc65 = function() {
        const ret = Symbol.iterator;
        return ret;
    };
    imports.wbg.__wbg_length_a446193dc22c12f8 = function(arg0) {
        const ret = arg0.length;
        return ret;
    };
    imports.wbg.__wbg_length_e2d2a49132c1b256 = function(arg0) {
        const ret = arg0.length;
        return ret;
    };
    imports.wbg.__wbg_new_405e22f390576ce2 = function() {
        const ret = new Object();
        return ret;
    };
    imports.wbg.__wbg_new_78feb108b6472713 = function() {
        const ret = new Array();
        return ret;
    };
    imports.wbg.__wbg_new_8a6f238a6ece86ea = function() {
        const ret = new Error();
        return ret;
    };
    imports.wbg.__wbg_new_a12002a7f91c75be = function(arg0) {
        const ret = new Uint8Array(arg0);
        return ret;
    };
    imports.wbg.__wbg_next_25feadfc0913fea9 = function(arg0) {
        const ret = arg0.next;
        return ret;
    };
    imports.wbg.__wbg_next_6574e1a8a62d1055 = function() { return handleError(function (arg0) {
        const ret = arg0.next();
        return ret;
    }, arguments) };
    imports.wbg.__wbg_set_37837023f3d740e8 = function(arg0, arg1, arg2) {
        arg0[arg1 >>> 0] = arg2;
    };
    imports.wbg.__wbg_set_3f1d0b984ed272ed = function(arg0, arg1, arg2) {
        arg0[arg1] = arg2;
    };
    imports.wbg.__wbg_set_65595bdd868b3009 = function(arg0, arg1, arg2) {
        arg0.set(arg1, arg2 >>> 0);
    };
    imports.wbg.__wbg_stack_0ed75d68575b0f3c = function(arg0, arg1) {
        const ret = arg1.stack;
        const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
        getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
    };
    imports.wbg.__wbg_value_cd1ffa7b1ab794f1 = function(arg0) {
        const ret = arg0.value;
        return ret;
    };
    imports.wbg.__wbg_wasmmoire2d_new = function(arg0) {
        const ret = WasmMoire2D.__wrap(arg0);
        return ret;
    };
    imports.wbg.__wbindgen_bigint_from_u64 = function(arg0) {
        const ret = BigInt.asUintN(64, arg0);
        return ret;
    };
    imports.wbg.__wbindgen_boolean_get = function(arg0) {
        const v = arg0;
        const ret = typeof(v) === 'boolean' ? (v ? 1 : 0) : 2;
        return ret;
    };
    imports.wbg.__wbindgen_debug_string = function(arg0, arg1) {
        const ret = debugString(arg1);
        const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
        getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
    };
    imports.wbg.__wbindgen_error_new = function(arg0, arg1) {
        const ret = new Error(getStringFromWasm0(arg0, arg1));
        return ret;
    };
    imports.wbg.__wbindgen_in = function(arg0, arg1) {
        const ret = arg0 in arg1;
        return ret;
    };
    imports.wbg.__wbindgen_init_externref_table = function() {
        const table = wasm.__wbindgen_export_4;
        const offset = table.grow(4);
        table.set(0, undefined);
        table.set(offset + 0, undefined);
        table.set(offset + 1, null);
        table.set(offset + 2, true);
        table.set(offset + 3, false);
        ;
    };
    imports.wbg.__wbindgen_is_function = function(arg0) {
        const ret = typeof(arg0) === 'function';
        return ret;
    };
    imports.wbg.__wbindgen_is_object = function(arg0) {
        const val = arg0;
        const ret = typeof(val) === 'object' && val !== null;
        return ret;
    };
    imports.wbg.__wbindgen_is_undefined = function(arg0) {
        const ret = arg0 === undefined;
        return ret;
    };
    imports.wbg.__wbindgen_jsval_loose_eq = function(arg0, arg1) {
        const ret = arg0 == arg1;
        return ret;
    };
    imports.wbg.__wbindgen_memory = function() {
        const ret = wasm.memory;
        return ret;
    };
    imports.wbg.__wbindgen_number_get = function(arg0, arg1) {
        const obj = arg1;
        const ret = typeof(obj) === 'number' ? obj : undefined;
        getDataViewMemory0().setFloat64(arg0 + 8 * 1, isLikeNone(ret) ? 0 : ret, true);
        getDataViewMemory0().setInt32(arg0 + 4 * 0, !isLikeNone(ret), true);
    };
    imports.wbg.__wbindgen_number_new = function(arg0) {
        const ret = arg0;
        return ret;
    };
    imports.wbg.__wbindgen_string_get = function(arg0, arg1) {
        const obj = arg1;
        const ret = typeof(obj) === 'string' ? obj : undefined;
        var ptr1 = isLikeNone(ret) ? 0 : passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        var len1 = WASM_VECTOR_LEN;
        getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
        getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
    };
    imports.wbg.__wbindgen_string_new = function(arg0, arg1) {
        const ret = getStringFromWasm0(arg0, arg1);
        return ret;
    };
    imports.wbg.__wbindgen_throw = function(arg0, arg1) {
        throw new Error(getStringFromWasm0(arg0, arg1));
    };

    return imports;
}

function __wbg_init_memory(imports, memory) {

}

function __wbg_finalize_init(instance, module) {
    wasm = instance.exports;
    __wbg_init.__wbindgen_wasm_module = module;
    cachedDataViewMemory0 = null;
    cachedFloat64ArrayMemory0 = null;
    cachedInt32ArrayMemory0 = null;
    cachedUint8ArrayMemory0 = null;


    wasm.__wbindgen_start();
    return wasm;
}

function initSync(module) {
    if (wasm !== undefined) return wasm;


    if (typeof module !== 'undefined') {
        if (Object.getPrototypeOf(module) === Object.prototype) {
            ({module} = module)
        } else {
            console.warn('using deprecated parameters for `initSync()`; pass a single object instead')
        }
    }

    const imports = __wbg_get_imports();

    __wbg_init_memory(imports);

    if (!(module instanceof WebAssembly.Module)) {
        module = new WebAssembly.Module(module);
    }

    const instance = new WebAssembly.Instance(module, imports);

    return __wbg_finalize_init(instance, module);
}

async function __wbg_init(module_or_path) {
    if (wasm !== undefined) return wasm;


    if (typeof module_or_path !== 'undefined') {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }

    if (typeof module_or_path === 'undefined') {
        module_or_path = new URL('moire_lattice_wasm_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports();

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    __wbg_init_memory(imports);

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module);
}

export { initSync };
export default __wbg_init;
