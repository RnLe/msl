let wasm;

let cachedUint8ArrayMemory0 = null;

function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

let cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });

cachedTextDecoder.decode();

const MAX_SAFARI_DECODE_BYTES = 2146435072;
let numBytesDecoded = 0;
function decodeText(ptr, len) {
    numBytesDecoded += len;
    if (numBytesDecoded >= MAX_SAFARI_DECODE_BYTES) {
        cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
        cachedTextDecoder.decode();
        numBytesDecoded = len;
    }
    return cachedTextDecoder.decode(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}

function getStringFromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return decodeText(ptr, len);
}

let WASM_VECTOR_LEN = 0;

const cachedTextEncoder = new TextEncoder();

if (!('encodeInto' in cachedTextEncoder)) {
    cachedTextEncoder.encodeInto = function (arg, view) {
        const buf = cachedTextEncoder.encode(arg);
        view.set(buf);
        return {
            read: arg.length,
            written: buf.length
        };
    }
}

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
        const ret = cachedTextEncoder.encodeInto(arg, view);

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

let cachedFloat64ArrayMemory0 = null;

function getFloat64ArrayMemory0() {
    if (cachedFloat64ArrayMemory0 === null || cachedFloat64ArrayMemory0.byteLength === 0) {
        cachedFloat64ArrayMemory0 = new Float64Array(wasm.memory.buffer);
    }
    return cachedFloat64ArrayMemory0;
}

function passArrayF64ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 8, 8) >>> 0;
    getFloat64ArrayMemory0().set(arg, ptr / 8);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function takeFromExternrefTable0(idx) {
    const value = wasm.__wbindgen_export_3.get(idx);
    wasm.__externref_table_dealloc(idx);
    return value;
}

function getArrayF64FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getFloat64ArrayMemory0().subarray(ptr / 8, ptr / 8 + len);
}

function _assertClass(instance, klass) {
    if (!(instance instanceof klass)) {
        throw new Error(`expected instance of ${klass.name}`);
    }
}
/**
 * Helper function to create a moiré lattice from a base lattice and transformation
 * This is an alternative to using the Moire2D constructor directly
 * @param {Lattice2D} base_lattice
 * @param {MoireTransformation} transformation
 * @returns {Moire2D}
 */
export function createMoireLattice(base_lattice, transformation) {
    _assertClass(base_lattice, Lattice2D);
    _assertClass(transformation, MoireTransformation);
    const ret = wasm.createMoireLattice(base_lattice.__wbg_ptr, transformation.__wbg_ptr);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return Moire2D.__wrap(ret[0]);
}

/**
 * Helper functions for creating standard lattices
 * @param {number} a
 * @returns {Lattice2D}
 */
export function create_square_lattice(a) {
    const ret = wasm.create_square_lattice(a);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return Lattice2D.__wrap(ret[0]);
}

/**
 * @param {number} a
 * @param {number} b
 * @returns {Lattice2D}
 */
export function create_rectangular_lattice(a, b) {
    const ret = wasm.create_rectangular_lattice(a, b);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return Lattice2D.__wrap(ret[0]);
}

/**
 * @param {number} a
 * @returns {Lattice2D}
 */
export function create_hexagonal_lattice(a) {
    const ret = wasm.create_hexagonal_lattice(a);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return Lattice2D.__wrap(ret[0]);
}

/**
 * @param {number} a
 * @param {number} b
 * @param {number} gamma
 * @returns {Lattice2D}
 */
export function create_oblique_lattice(a, b, gamma) {
    const ret = wasm.create_oblique_lattice(a, b, gamma);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return Lattice2D.__wrap(ret[0]);
}

export function init_panic_hook() {
    wasm.init_panic_hook();
}

/**
 * Version information
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
 * 2D Bravais lattice classification (WASM-compatible)
 * @enum {0 | 1 | 2 | 3 | 4}
 */
export const Bravais2D = Object.freeze({
    Square: 0, "0": "Square",
    Rectangular: 1, "1": "Rectangular",
    CenteredRectangular: 2, "2": "CenteredRectangular",
    Hexagonal: 3, "3": "Hexagonal",
    Oblique: 4, "4": "Oblique",
});

const BaseMatrixDirectFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_basematrixdirect_free(ptr >>> 0, 1));
/**
 * WASM-compatible wrapper for 2D base matrix operations
 */
export class BaseMatrixDirect {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        BaseMatrixDirectFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_basematrixdirect_free(ptr, 0);
    }
    /**
     * Create a 2D base matrix from two base vectors
     * Each vector should be provided as a 3-element array [x, y, z]
     * @param {Float64Array} base_1
     * @param {Float64Array} base_2
     */
    constructor(base_1, base_2) {
        const ptr0 = passArrayF64ToWasm0(base_1, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF64ToWasm0(base_2, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.basematrixdirect_new(ptr0, len0, ptr1, len1);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0] >>> 0;
        BaseMatrixDirectFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Get the base matrix as a flat array (column-major order)
     * @returns {Float64Array}
     */
    getMatrix() {
        const ret = wasm.basematrixdirect_getMatrix(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * Get the determinant of the base matrix
     * @returns {number}
     */
    determinant() {
        const ret = wasm.basematrixdirect_determinant(this.__wbg_ptr);
        return ret;
    }
}
if (Symbol.dispose) BaseMatrixDirect.prototype[Symbol.dispose] = BaseMatrixDirect.prototype.free;

const BaseMatrixReciprocalFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_basematrixreciprocal_free(ptr >>> 0, 1));
/**
 * WASM-compatible wrapper for reciprocal space base matrix
 */
export class BaseMatrixReciprocal {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        BaseMatrixReciprocalFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_basematrixreciprocal_free(ptr, 0);
    }
    /**
     * Get the base matrix as a flat array (column-major order)
     * @returns {Float64Array}
     */
    getMatrix() {
        const ret = wasm.basematrixreciprocal_getMatrix(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * Get the determinant of the base matrix
     * @returns {number}
     */
    determinant() {
        const ret = wasm.basematrixdirect_determinant(this.__wbg_ptr);
        return ret;
    }
}
if (Symbol.dispose) BaseMatrixReciprocal.prototype[Symbol.dispose] = BaseMatrixReciprocal.prototype.free;

const Lattice2DFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_lattice2d_free(ptr >>> 0, 1));
/**
 * WASM-compatible wrapper for 2D lattice
 */
export class Lattice2D {

    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(Lattice2D.prototype);
        obj.__wbg_ptr = ptr;
        Lattice2DFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        Lattice2DFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_lattice2d_free(ptr, 0);
    }
    /**
     * Create a 2D lattice from direct space basis vectors
     * The matrix should be provided as a flat array in column-major order (9 elements)
     * @param {Float64Array} direct_matrix
     */
    constructor(direct_matrix) {
        const ptr0 = passArrayF64ToWasm0(direct_matrix, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.lattice2d_new(ptr0, len0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0] >>> 0;
        Lattice2DFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Get the direct space basis matrix as a flat array (column-major order)
     * @returns {Float64Array}
     */
    getDirectBasis() {
        const ret = wasm.lattice2d_getDirectBasis(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * Get the reciprocal space basis matrix as a flat array (column-major order)
     * @returns {Float64Array}
     */
    getReciprocalBasis() {
        const ret = wasm.lattice2d_getReciprocalBasis(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * Get the Bravais lattice type
     * @returns {Bravais2D}
     */
    getBravaisType() {
        const ret = wasm.lattice2d_getBravaisType(this.__wbg_ptr);
        return ret;
    }
    /**
     * Get the Wigner-Seitz cell vertices as a flat array
     * Each vertex is [x, y, z], returned as a flat array
     * @returns {Float64Array}
     */
    getWignerSeitzVertices() {
        const ret = wasm.lattice2d_getWignerSeitzVertices(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * Get the Brillouin zone vertices as a flat array
     * Each vertex is [x, y, z], returned as a flat array
     * @returns {Float64Array}
     */
    getBrillouinZoneVertices() {
        const ret = wasm.lattice2d_getBrillouinZoneVertices(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * Generate direct lattice points in a rectangle
     * @param {number} width
     * @param {number} height
     * @returns {Float64Array}
     */
    getDirectLatticePoints(width, height) {
        const ret = wasm.lattice2d_getDirectLatticePoints(this.__wbg_ptr, width, height);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * Generate reciprocal lattice points in a rectangle
     * @param {number} width
     * @param {number} height
     * @returns {Float64Array}
     */
    getReciprocalLatticePoints(width, height) {
        const ret = wasm.lattice2d_getReciprocalLatticePoints(this.__wbg_ptr, width, height);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * Generate high symmetry k-path for band structure calculations
     * @param {number} n_points_per_segment
     * @returns {Float64Array}
     */
    getHighSymmetryPath(n_points_per_segment) {
        const ret = wasm.lattice2d_getHighSymmetryPath(this.__wbg_ptr, n_points_per_segment);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * Check if a point is in the Brillouin zone
     * @param {Float64Array} k_point
     * @returns {boolean}
     */
    isInBrillouinZone(k_point) {
        const ptr0 = passArrayF64ToWasm0(k_point, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.lattice2d_isInBrillouinZone(this.__wbg_ptr, ptr0, len0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0] !== 0;
    }
    /**
     * Reduce a k-point to the first Brillouin zone
     * @param {Float64Array} k_point
     * @returns {Float64Array}
     */
    reduceToBrillouinZone(k_point) {
        const ptr0 = passArrayF64ToWasm0(k_point, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.lattice2d_reduceToBrillouinZone(this.__wbg_ptr, ptr0, len0);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v2;
    }
}
if (Symbol.dispose) Lattice2D.prototype[Symbol.dispose] = Lattice2D.prototype.free;

const Moire2DFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_moire2d_free(ptr >>> 0, 1));
/**
 * WASM-compatible wrapper for Moire2D
 */
export class Moire2D {

    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(Moire2D.prototype);
        obj.__wbg_ptr = ptr;
        Moire2DFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        Moire2DFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_moire2d_free(ptr, 0);
    }
    /**
     * Create a Moiré lattice from a base lattice and transformation
     * @param {Lattice2D} base_lattice
     * @param {MoireTransformation} transformation
     */
    constructor(base_lattice, transformation) {
        _assertClass(base_lattice, Lattice2D);
        _assertClass(transformation, MoireTransformation);
        const ret = wasm.moire2d_new(base_lattice.__wbg_ptr, transformation.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0] >>> 0;
        Moire2DFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Get the effective moiré lattice direct space basis matrix
     * @returns {Float64Array}
     */
    getEffectiveDirectBasis() {
        const ret = wasm.moire2d_getEffectiveDirectBasis(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * Get the effective moiré lattice reciprocal space basis matrix
     * @returns {Float64Array}
     */
    getEffectiveReciprocalBasis() {
        const ret = wasm.moire2d_getEffectiveReciprocalBasis(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * Get the Bravais lattice type of the effective moiré lattice
     * @returns {Bravais2D}
     */
    getEffectiveBravaisType() {
        const ret = wasm.moire2d_getEffectiveBravaisType(this.__wbg_ptr);
        return ret;
    }
    /**
     * Get the Wigner-Seitz cell vertices of the effective lattice
     * @returns {Float64Array}
     */
    getEffectiveWignerSeitzVertices() {
        const ret = wasm.moire2d_getEffectiveWignerSeitzVertices(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * Get the Brillouin zone vertices of the effective lattice
     * @returns {Float64Array}
     */
    getEffectiveBrillouinZoneVertices() {
        const ret = wasm.moire2d_getEffectiveBrillouinZoneVertices(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * Generate direct lattice points in a rectangle for the effective moiré lattice
     * @param {number} width
     * @param {number} height
     * @returns {Float64Array}
     */
    getEffectiveDirectLatticePoints(width, height) {
        const ret = wasm.moire2d_getEffectiveDirectLatticePoints(this.__wbg_ptr, width, height);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * Generate reciprocal lattice points in a rectangle for the effective moiré lattice
     * @param {number} width
     * @param {number} height
     * @returns {Float64Array}
     */
    getEffectiveReciprocalLatticePoints(width, height) {
        const ret = wasm.moire2d_getEffectiveReciprocalLatticePoints(this.__wbg_ptr, width, height);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * Generate high symmetry k-path for the effective moiré lattice
     * @param {number} n_points_per_segment
     * @returns {Float64Array}
     */
    getEffectiveHighSymmetryPath(n_points_per_segment) {
        const ret = wasm.moire2d_getEffectiveHighSymmetryPath(this.__wbg_ptr, n_points_per_segment);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * Check if a point is in the Brillouin zone of the effective lattice
     * @param {Float64Array} k_point
     * @returns {boolean}
     */
    isInEffectiveBrillouinZone(k_point) {
        const ptr0 = passArrayF64ToWasm0(k_point, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.moire2d_isInEffectiveBrillouinZone(this.__wbg_ptr, ptr0, len0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0] !== 0;
    }
    /**
     * Reduce a k-point to the first Brillouin zone of the effective lattice
     * @param {Float64Array} k_point
     * @returns {Float64Array}
     */
    reduceToEffectiveBrillouinZone(k_point) {
        const ptr0 = passArrayF64ToWasm0(k_point, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.moire2d_reduceToEffectiveBrillouinZone(this.__wbg_ptr, ptr0, len0);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v2;
    }
    /**
     * Get the first constituent lattice
     * @returns {Lattice2D}
     */
    getLattice1() {
        const ret = wasm.moire2d_getLattice1(this.__wbg_ptr);
        return Lattice2D.__wrap(ret);
    }
    /**
     * Get the second constituent lattice
     * @returns {Lattice2D}
     */
    getLattice2() {
        const ret = wasm.moire2d_getLattice2(this.__wbg_ptr);
        return Lattice2D.__wrap(ret);
    }
    /**
     * Get the transformation that was applied
     * @returns {MoireTransformation}
     */
    getTransformation() {
        const ret = wasm.moire2d_getTransformation(this.__wbg_ptr);
        return MoireTransformation.__wrap(ret);
    }
    /**
     * Check if the moiré lattice is commensurate
     * @returns {boolean}
     */
    isCommensurate() {
        const ret = wasm.moire2d_isCommensurate(this.__wbg_ptr);
        return ret !== 0;
    }
}
if (Symbol.dispose) Moire2D.prototype[Symbol.dispose] = Moire2D.prototype.free;

const MoireTransformationFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_moiretransformation_free(ptr >>> 0, 1));
/**
 * WASM-compatible wrapper for MoireTransformation
 */
export class MoireTransformation {

    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(MoireTransformation.prototype);
        obj.__wbg_ptr = ptr;
        MoireTransformationFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        MoireTransformationFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_moiretransformation_free(ptr, 0);
    }
    /**
     * Create a simple twist (rotation) transformation
     * @param {number} angle
     */
    constructor(angle) {
        const ret = wasm.moiretransformation_new_twist(angle);
        this.__wbg_ptr = ret >>> 0;
        MoireTransformationFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Create a rotation and scaling transformation
     * @param {number} angle
     * @param {number} scale
     * @returns {MoireTransformation}
     */
    static newRotationScale(angle, scale) {
        const ret = wasm.moiretransformation_newRotationScale(angle, scale);
        return MoireTransformation.__wrap(ret);
    }
    /**
     * Create an anisotropic scaling transformation
     * @param {number} scale_x
     * @param {number} scale_y
     * @returns {MoireTransformation}
     */
    static newAnisotropicScale(scale_x, scale_y) {
        const ret = wasm.moiretransformation_newAnisotropicScale(scale_x, scale_y);
        return MoireTransformation.__wrap(ret);
    }
    /**
     * Create a shear transformation
     * @param {number} shear_x
     * @param {number} shear_y
     * @returns {MoireTransformation}
     */
    static newShear(shear_x, shear_y) {
        const ret = wasm.moiretransformation_newShear(shear_x, shear_y);
        return MoireTransformation.__wrap(ret);
    }
    /**
     * Create a general matrix transformation
     * Matrix should be provided as a flat array in column-major order (4 elements for 2x2)
     * @param {Float64Array} matrix
     * @returns {MoireTransformation}
     */
    static newGeneral(matrix) {
        const ptr0 = passArrayF64ToWasm0(matrix, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.moiretransformation_newGeneral(ptr0, len0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return MoireTransformation.__wrap(ret[0]);
    }
    /**
     * Get the transformation matrix as a flat array (column-major order, 4 elements)
     * @returns {Float64Array}
     */
    getMatrix2() {
        const ret = wasm.moiretransformation_getMatrix2(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * Get the transformation matrix as a 3x3 matrix (embedding 2D in 3D)
     * Returns a flat array in column-major order (9 elements)
     * @returns {Float64Array}
     */
    getMatrix3() {
        const ret = wasm.moiretransformation_getMatrix3(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
}
if (Symbol.dispose) MoireTransformation.prototype[Symbol.dispose] = MoireTransformation.prototype.free;

const PolyhedronFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_polyhedron_free(ptr >>> 0, 1));
/**
 * WASM-compatible wrapper for Polyhedron
 */
export class Polyhedron {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        PolyhedronFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_polyhedron_free(ptr, 0);
    }
    /**
     * Get the vertices of the polyhedron as a flat array
     * Each vertex is represented as [x, y, z]
     * @returns {Float64Array}
     */
    getVertices() {
        const ret = wasm.polyhedron_getVertices(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * Get the number of vertices
     * @returns {number}
     */
    vertexCount() {
        const ret = wasm.polyhedron_vertexCount(this.__wbg_ptr);
        return ret >>> 0;
    }
}
if (Symbol.dispose) Polyhedron.prototype[Symbol.dispose] = Polyhedron.prototype.free;

const EXPECTED_RESPONSE_TYPES = new Set(['basic', 'cors', 'default']);

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);

            } catch (e) {
                const validResponse = module.ok && EXPECTED_RESPONSE_TYPES.has(module.type);

                if (validResponse && module.headers.get('Content-Type') !== 'application/wasm') {
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
    imports.wbg.__wbg_new_8a6f238a6ece86ea = function() {
        const ret = new Error();
        return ret;
    };
    imports.wbg.__wbg_stack_0ed75d68575b0f3c = function(arg0, arg1) {
        const ret = arg1.stack;
        const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
        getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
    };
    imports.wbg.__wbg_wbindgenthrow_451ec1a8469d7eb6 = function(arg0, arg1) {
        throw new Error(getStringFromWasm0(arg0, arg1));
    };
    imports.wbg.__wbindgen_cast_2241b6af4c4b2941 = function(arg0, arg1) {
        // Cast intrinsic for `Ref(String) -> Externref`.
        const ret = getStringFromWasm0(arg0, arg1);
        return ret;
    };
    imports.wbg.__wbindgen_init_externref_table = function() {
        const table = wasm.__wbindgen_export_3;
        const offset = table.grow(4);
        table.set(0, undefined);
        table.set(offset + 0, undefined);
        table.set(offset + 1, null);
        table.set(offset + 2, true);
        table.set(offset + 3, false);
        ;
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
