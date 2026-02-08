'use client';

import React, { useState, useEffect, useMemo } from 'react';
import { motion } from 'framer-motion';
import { RefreshCcw, Info, MousePointer2, Maximize2, MoveDiagonal, Play, ArrowRight } from 'lucide-react';

// --- Type Definitions ---
type Matrix = {
  a: number;
  b: number;
  c: number;
  d: number;
};

type Vector = {
  x: number;
  y: number;
};

// --- Constants ---
const GRID_SIZE = 10; // Grid range (-10 ~ 10)
const GRID_STEP = 1;  // Grid step
const VIEWBOX_SIZE = 12; // Viewbox size

// --- Helper Functions ---

// Convert math coordinates (x, y) to SVG coordinates (x, y) (flip Y axis)
const toSvg = (x: number, y: number) => ({ x: x, y: -y });

// Calculate matrix-vector product
const transformVector = (matrix: Matrix, vector: Vector): Vector => {
  return {
    x: matrix.a * vector.x + matrix.b * vector.y,
    y: matrix.c * vector.x + matrix.d * vector.y,
  };
};

// Calculate matrix multiplication (C = B * A)
const multiplyMatrices = (m1: Matrix, m2: Matrix): Matrix => {
  return {
    a: m1.a * m2.a + m1.b * m2.c,
    b: m1.a * m2.b + m1.b * m2.d,
    c: m1.c * m2.a + m1.d * m2.c,
    d: m1.c * m2.b + m1.d * m2.d,
  };
};

// Calculate inverse matrix
const invertMatrix = (m: Matrix): Matrix | null => {
  const det = m.a * m.d - m.b * m.c;
  // If determinant is close to 0, inverse matrix does not exist
  if (Math.abs(det) < 1e-6) return null;
  const invDet = 1 / det;
  return {
    a: m.d * invDet,
    b: -m.b * invDet,
    c: -m.c * invDet,
    d: m.a * invDet,
  };
};

// Analyze matrix properties and return explanatory text
const analyzeMatrix = (m: Matrix): { title: string; description: string } => {
  const det = m.a * m.d - m.b * m.c;
  const epsilon = 0.001;
  const isClose = (v1: number, v2: number) => Math.abs(v1 - v2) < epsilon;

  if (isClose(det, 0)) {
    return {
      title: 'Singular Matrix',
      description: 'Determinant is 0. The space collapses into a line or a point, making inverse transformation impossible.',
    };
  }

  if (isClose(m.a, 1) && isClose(m.b, 0) && isClose(m.c, 0) && isClose(m.d, 1)) {
    return {
      title: 'Identity Matrix',
      description: 'A transformation that changes nothing.',
    };
  }

  // Rotation matrix check: [[cos, -sin], [sin, cos]]
  // a = d, b = -c, a^2 + c^2 = 1 (if no scaling)
  // Simple check if rotation component is included
  if (isClose(m.a, m.d) && isClose(m.b, -m.c) && !isClose(m.b, 0)) {
    const angle = Math.atan2(m.c, m.a) * (180 / Math.PI);
    return {
      title: 'Rotation',
      description: `Rotates approximately ${Math.round(angle)} degrees around the origin (may include scaling).`,
    };
  }

  // Scaling: b=0, c=0
  if (isClose(m.b, 0) && isClose(m.c, 0)) {
    if (isClose(m.a, m.d)) {
      return {
        title: 'Uniform Scaling',
        description: `Scales the entire space by ${m.a.toFixed(2)}x.`,
      };
    }
    return {
      title: 'Non-uniform Scaling',
      description: `Scales by ${m.a}x along the X-axis and ${m.d}x along the Y-axis.`,
    };
  }

  // Shear
  // Shear X: a=1, c=0, d=1, b!=0
  if (isClose(m.a, 1) && isClose(m.c, 0) && isClose(m.d, 1) && !isClose(m.b, 0)) {
    return {
      title: 'Shear X',
      description: 'Shifts parallel to the X-axis (X changes depending on Y coordinate).',
    };
  }
  // Shear Y: a=1, b=0, d=1, c!=0
  if (isClose(m.a, 1) && isClose(m.b, 0) && isClose(m.d, 1) && !isClose(m.c, 0)) {
    return {
      title: 'Shear Y',
      description: 'Shifts parallel to the Y-axis (Y changes depending on X coordinate).',
    };
  }

  return {
    title: 'General Linear Transformation',
    description: `Determinant: ${det.toFixed(2)}. Basis vectors i and j move to new positions.`,
  };
};

// Calculate Eigenvalues and Eigenvectors
const calculateEigen = (m: Matrix): { values: number[]; vectors: Vector[] } | null => {
  const trace = m.a + m.d;
  const det = m.a * m.d - m.b * m.c;
  const discriminant = trace * trace - 4 * det;

  // If discriminant is negative, complex solutions (not visualized here)
  if (discriminant < -0.0001) return null;

  const sqrtD = Math.sqrt(Math.max(0, discriminant));
  const l1 = (trace + sqrtD) / 2;
  const l2 = (trace - sqrtD) / 2;

  // Calculate eigenvector (A - lambda I)v = 0
  const getEigenVector = (lambda: number): Vector => {
    // Special handling for diagonal matrix (b=0, c=0)
    if (Math.abs(m.b) < 1e-6 && Math.abs(m.c) < 1e-6) {
      // If lambda is close to a, return (1,0), if close to d, return (0,1)
      // If repeated root (a=d), separated in subsequent processing, but return basic axis here
      if (Math.abs(lambda - m.a) < 1e-6) return { x: 1, y: 0 };
      return { x: 0, y: 1 };
    }

    // General form:
    // If b != 0, x=b, y=lambda-a
    if (Math.abs(m.b) > 1e-6) {
      return { x: m.b, y: lambda - m.a };
    }
    // If c != 0, x=lambda-d, y=c
    if (Math.abs(m.c) > 1e-6) {
      return { x: lambda - m.d, y: m.c };
    }
    
    return { x: 0, y: 0 };
  };

  let v1 = getEigenVector(l1);
  let v2 = getEigenVector(l2);

  // If diagonal matrix and repeated root (e.g., identity matrix), force orthogonal basis assignment
  if (Math.abs(m.b) < 1e-6 && Math.abs(m.c) < 1e-6 && Math.abs(l1 - l2) < 1e-6) {
      v1 = { x: 1, y: 0 };
      v2 = { x: 0, y: 1 };
  }

  // Normalize (make length 1)
  const normalize = (v: Vector) => {
    const len = Math.sqrt(v.x * v.x + v.y * v.y);
    return len < 1e-6 ? { x: 0, y: 0 } : { x: v.x / len, y: v.y / len };
  };

  return {
    values: [l1, l2],
    vectors: [normalize(v1), normalize(v2)],
  };
};

// Calculate Singular Value Decomposition (SVD): A = U * Sigma * V^T
const calculateSVD = (m: Matrix) => {
  // 1. Calculate A^T * A (becomes symmetric matrix)
  const ata = {
    a: m.a * m.a + m.c * m.c,
    b: m.a * m.b + m.c * m.d,
    c: m.a * m.b + m.c * m.d,
    d: m.b * m.b + m.d * m.d
  };

  // 2. Calculate eigenvalues/eigenvectors of A^T * A -> V and Sigma^2
  const eigen = calculateEigen(ata);
  
  // Default values (if calculation fails)
  let v1 = { x: 1, y: 0 };
  let v2 = { x: 0, y: 1 };
  let s1 = 0;
  let s2 = 0;

  if (eigen) {
    // Sort eigenvalues in descending order (s1 >= s2)
    if (eigen.values[0] < eigen.values[1]) {
      s1 = Math.sqrt(Math.max(0, eigen.values[1]));
      s2 = Math.sqrt(Math.max(0, eigen.values[0]));
      v1 = eigen.vectors[1];
      v2 = eigen.vectors[0];
    } else {
      s1 = Math.sqrt(Math.max(0, eigen.values[0]));
      s2 = Math.sqrt(Math.max(0, eigen.values[1]));
      v1 = eigen.vectors[0];
      v2 = eigen.vectors[1];
    }
  }

  // 3. Construct V and V^T
  const Vt = { a: v1.x, b: v1.y, c: v2.x, d: v2.y }; // Arrange eigenvectors in rows

  // 4. Calculate U (u_i = A * v_i / s_i)
  const applyA = (v: Vector) => transformVector(m, v);
  
  let u1: Vector, u2: Vector;
  
  // If singular value is close to 0, choose arbitrary orthonormal basis
  if (s1 > 1e-6) {
    const av1 = applyA(v1);
    u1 = { x: av1.x / s1, y: av1.y / s1 };
  } else {
    u1 = { x: 1, y: 0 };
  }

  if (s2 > 1e-6) {
    const av2 = applyA(v2);
    u2 = { x: av2.x / s2, y: av2.y / s2 };
  } else {
    u2 = { x: -u1.y, y: u1.x }; // Make orthogonal to u1
  }

  const U = { a: u1.x, b: u2.x, c: u1.y, d: u2.y };
  const Sigma = { a: s1, b: 0, c: 0, d: s2 };

  return { U, Sigma, Vt, s1, s2 };
};

// --- Components ---

export default function LinearAlgebraPage() {
  const [matrixA, setMatrixA] = useState<Matrix>({ a: 1, b: 0, c: 0, d: 1 });
  const [matrixB, setMatrixB] = useState<Matrix>({ a: 1, b: 0, c: 0, d: 1 });
  const [animationStep, setAnimationStep] = useState<'idle' | 'stepA' | 'stepBA'>('stepBA');
  const [svdStep, setSvdStep] = useState<0 | 1 | 2 | 3>(0); // 0:off, 1:Vt, 2:Sigma*Vt, 3:U*Sigma*Vt
  const [hoveredVector, setHoveredVector] = useState<'i' | 'j' | null>(null);

  // Generate grid lines
  const gridLines = useMemo(() => {
    const lines = [];
    // Vertical lines (x = k)
    for (let x = -GRID_SIZE; x <= GRID_SIZE; x += GRID_STEP) {
      lines.push({
        key: `v${x}`,
        start: { x, y: -GRID_SIZE },
        end: { x, y: GRID_SIZE },
        isAxis: x === 0,
      });
    }
    // Horizontal lines (y = k)
    for (let y = -GRID_SIZE; y <= GRID_SIZE; y += GRID_STEP) {
      lines.push({
        key: `h${y}`,
        start: { x: -GRID_SIZE, y },
        end: { x: GRID_SIZE, y },
        isAxis: y === 0,
      });
    }
    return lines;
  }, []);

  // Generate dot grid
  const dots = useMemo(() => {
    const points = [];
    for (let x = -GRID_SIZE; x <= GRID_SIZE; x += GRID_STEP) {
      for (let y = -GRID_SIZE; y <= GRID_SIZE; y += GRID_STEP) {
        // Exclude origin as it is drawn separately
        if (x === 0 && y === 0) continue;
        points.push({ x, y });
      }
    }
    return points;
  }, []);

  // Calculate composite matrix (C = B * A)
  const matrixC = useMemo(() => multiplyMatrices(matrixB, matrixA), [matrixA, matrixB]);

  // Calculate SVD (for Matrix A)
  const svdData = useMemo(() => calculateSVD(matrixA), [matrixA]);

  // Current matrix to display (for animation)
  const currentMatrix = useMemo(() => {
    // If SVD animation is active
    if (svdStep > 0) {
      if (svdStep === 1) return svdData.Vt; // Step 1: Rotation (V^T)
      if (svdStep === 2) return multiplyMatrices(svdData.Sigma, svdData.Vt); // Step 2: Scaling (Sigma * V^T)
      if (svdStep === 3) return matrixA; // Step 3: Rotation (U * Sigma * V^T = A)
    }

    // Normal composite transformation animation
    switch (animationStep) {
      case 'idle': // Initial state (Identity matrix)
        return { a: 1, b: 0, c: 0, d: 1 };
      case 'stepA': // Transformation by A
        return matrixA;
      case 'stepBA': // Transformation by B(A) (Final result)
        return matrixC;
      default:
        return matrixC;
    }
  }, [animationStep, svdStep, matrixA, matrixC, svdData]);

  // Transformed basis vectors
  const transformedI = transformVector(currentMatrix, { x: 1, y: 0 });
  const transformedJ = transformVector(currentMatrix, { x: 0, y: 1 });

  // 4th point of parallelogram and determinant (area)
  const transformedSum = {
    x: transformedI.x + transformedJ.x,
    y: transformedI.y + transformedJ.y,
  };
  const det = currentMatrix.a * currentMatrix.d - currentMatrix.b * currentMatrix.c;
  const originSvg = toSvg(0, 0);
  const iSvg = toSvg(transformedI.x, transformedI.y);
  const sumSvg = toSvg(transformedSum.x, transformedSum.y);
  const jSvg = toSvg(transformedJ.x, transformedJ.y);
  const centerSvg = toSvg(transformedSum.x / 2, transformedSum.y / 2);

  // Analysis result
  const analysis = analyzeMatrix(currentMatrix);
  
  // Calculate Eigenvalues/Eigenvectors
  const eigenData = useMemo(() => calculateEigen(currentMatrix), [currentMatrix]);

  // Calculate Eigendecomposition (A = PDP^-1)
  const diagonalization = useMemo(() => {
    if (!eigenData) return null;
    
    const [l1, l2] = eigenData.values;
    const [v1, v2] = eigenData.vectors;
    
    // P = [v1 v2] (Arrange eigenvectors in columns)
    const P: Matrix = {
      a: v1.x, b: v2.x,
      c: v1.y, d: v2.y
    };
    
    // D = diag(l1, l2) (Arrange eigenvalues on diagonal)
    const D: Matrix = {
      a: l1, b: 0,
      c: 0, d: l2
    };
    
    // Inverse of P
    const Pinv = invertMatrix(P);
    
    return { P, D, Pinv };
  }, [eigenData]);

  // Generate paths for unit circle and transformed ellipse
  const { unitCirclePath, transformedEllipsePath } = useMemo(() => {
    const steps = 90; // 4 degree steps
    let unitPath = "";
    let ellipsePath = "";

    for (let i = 0; i <= steps; i++) {
      const theta = (i / steps) * Math.PI * 2;
      const cos = Math.cos(theta);
      const sin = Math.sin(theta);
      
      // Unit circle (radius 1)
      const p = toSvg(cos, sin);
      unitPath += `${i === 0 ? "M" : "L"} ${p.x} ${p.y}`;

      // Transformed ellipse
      const t = transformVector(currentMatrix, { x: cos, y: sin });
      const tp = toSvg(t.x, t.y);
      ellipsePath += `${i === 0 ? "M" : "L"} ${tp.x} ${tp.y}`;
    }

    return {
      unitCirclePath: unitPath + " Z",
      transformedEllipsePath: ellipsePath + " Z"
    };
  }, [currentMatrix]);

  // Input handler
  const handleInputChange = (target: 'A' | 'B', key: keyof Matrix, value: string) => {
    const num = parseFloat(value);
    if (!isNaN(num)) {
      if (target === 'A') {
        setMatrixA((prev) => ({ ...prev, [key]: num }));
      } else {
        setMatrixB((prev) => ({ ...prev, [key]: num }));
      }
      // Show final result during editing
      setAnimationStep('stepBA');
      setSvdStep(0);
    } else if (value === '' || value === '-') {
       // Allow partial input...
    }
  };

  // Apply inverse transformation handler
  const detA = matrixA.a * matrixA.d - matrixA.b * matrixA.c;
  const isSingularA = Math.abs(detA) < 1e-4;

  const handleApplyInverse = () => {
    const inv = invertMatrix(matrixA);
    if (inv) {
      setMatrixB(inv);
      setAnimationStep('stepBA');
      setSvdStep(0);
    }
  };

  // Apply preset
  const applyPreset = (type: 'identity' | 'rotate90' | 'scale2' | 'shearX') => {
    // Reset B and apply to A when applying preset
    setMatrixB({ a: 1, b: 0, c: 0, d: 1 });
    setAnimationStep('stepBA');
    setSvdStep(0);

    switch (type) {
      case 'identity':
        setMatrixA({ a: 1, b: 0, c: 0, d: 1 });
        break;
      case 'rotate90':
        setMatrixA({ a: 0, b: -1, c: 1, d: 0 });
        break;
      case 'scale2':
        setMatrixA({ a: 2, b: 0, c: 0, d: 2 });
        break;
      case 'shearX':
        setMatrixA({ a: 1, b: 1, c: 0, d: 1 });
        break;
    }
  };

  // Play animation
  const playAnimation = () => {
    setSvdStep(0);
    setAnimationStep('idle');
    setTimeout(() => setAnimationStep('stepA'), 800);
    setTimeout(() => setAnimationStep('stepBA'), 2000);
  };

  // Play SVD animation
  const playSVD = () => {
    setAnimationStep('idle'); // Reset normal animation
    setSvdStep(0);
    setTimeout(() => setSvdStep(1), 500);  // V^T
    setTimeout(() => setSvdStep(2), 2000); // Sigma
    setTimeout(() => setSvdStep(3), 3500); // U
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 p-4 md:p-8 font-sans">
      <div className="max-w-6xl mx-auto grid grid-cols-1 lg:grid-cols-3 gap-8">
        
        {/* Left column: Control panel */}
        <div className="lg:col-span-1 space-y-6">
          <header>
            <h1 className="text-2xl font-bold text-slate-800 mb-2">Linear Transformation Visualizer</h1>
            <p className="text-slate-600 text-sm">
              Observe the composite transformation ($C = BA$) where matrix $A$ is applied first, followed by matrix $B$.
            </p>
          </header>

          {/* Matrix Input A */}
          <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
            <h2 className="text-sm font-semibold text-slate-500 uppercase tracking-wider mb-4 flex items-center gap-2">
              1. Transformation Matrix A (Applied First)
            </h2>
            <div className="flex items-center justify-center gap-4 text-2xl font-mono">
              <span className="text-slate-400 select-none">{'('}</span>
              <div className="grid grid-cols-2 gap-x-4 gap-y-2">
                <div className="flex flex-col">
                  <input
                    type="number"
                    step="0.1"
                    value={matrixA.a}
                    onChange={(e) => handleInputChange('A', 'a', e.target.value)}
                    className="w-16 h-12 text-center bg-slate-100 rounded-md border border-transparent focus:border-blue-500 focus:bg-white focus:ring-2 focus:ring-blue-200 outline-none transition-all"
                  />
                  <label className="text-xs text-center text-slate-400 mt-1">a</label>
                </div>
                <div className="flex flex-col">
                  <input
                    type="number"
                    step="0.1"
                    value={matrixA.b}
                    onChange={(e) => handleInputChange('A', 'b', e.target.value)}
                    className="w-16 h-12 text-center bg-slate-100 rounded-md border border-transparent focus:border-blue-500 focus:bg-white focus:ring-2 focus:ring-blue-200 outline-none transition-all"
                  />
                  <label className="text-xs text-center text-slate-400 mt-1">b</label>
                </div>
                <div className="flex flex-col">
                  <input
                    type="number"
                    step="0.1"
                    value={matrixA.c}
                    onChange={(e) => handleInputChange('A', 'c', e.target.value)}
                    className="w-16 h-12 text-center bg-slate-100 rounded-md border border-transparent focus:border-blue-500 focus:bg-white focus:ring-2 focus:ring-blue-200 outline-none transition-all"
                  />
                  <label className="text-xs text-center text-slate-400 mt-1">c</label>
                </div>
                <div className="flex flex-col">
                  <input
                    type="number"
                    step="0.1"
                    value={matrixA.d}
                    onChange={(e) => handleInputChange('A', 'd', e.target.value)}
                    className="w-16 h-12 text-center bg-slate-100 rounded-md border border-transparent focus:border-blue-500 focus:bg-white focus:ring-2 focus:ring-blue-200 outline-none transition-all"
                  />
                  <label className="text-xs text-center text-slate-400 mt-1">d</label>
                </div>
              </div>
              <span className="text-slate-400 select-none">{')'}</span>
            </div>
          </div>

          {/* Matrix Input B */}
          <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
            <h2 className="text-sm font-semibold text-slate-500 uppercase tracking-wider mb-4 flex items-center gap-2">
              2. Transformation Matrix B (Applied Next)
            </h2>
            <div className="flex items-center justify-center gap-4 text-2xl font-mono">
              <span className="text-slate-400 select-none">{'('}</span>
              <div className="grid grid-cols-2 gap-x-4 gap-y-2">
                <div className="flex flex-col">
                  <input
                    type="number"
                    step="0.1"
                    value={matrixB.a}
                    onChange={(e) => handleInputChange('B', 'a', e.target.value)}
                    className="w-16 h-12 text-center bg-slate-100 rounded-md border border-transparent focus:border-purple-500 focus:bg-white focus:ring-2 focus:ring-purple-200 outline-none transition-all"
                  />
                </div>
                <div className="flex flex-col">
                  <input
                    type="number"
                    step="0.1"
                    value={matrixB.b}
                    onChange={(e) => handleInputChange('B', 'b', e.target.value)}
                    className="w-16 h-12 text-center bg-slate-100 rounded-md border border-transparent focus:border-purple-500 focus:bg-white focus:ring-2 focus:ring-purple-200 outline-none transition-all"
                  />
                </div>
                <div className="flex flex-col">
                  <input
                    type="number"
                    step="0.1"
                    value={matrixB.c}
                    onChange={(e) => handleInputChange('B', 'c', e.target.value)}
                    className="w-16 h-12 text-center bg-slate-100 rounded-md border border-transparent focus:border-purple-500 focus:bg-white focus:ring-2 focus:ring-purple-200 outline-none transition-all"
                  />
                </div>
                <div className="flex flex-col">
                  <input
                    type="number"
                    step="0.1"
                    value={matrixB.d}
                    onChange={(e) => handleInputChange('B', 'd', e.target.value)}
                    className="w-16 h-12 text-center bg-slate-100 rounded-md border border-transparent focus:border-purple-500 focus:bg-white focus:ring-2 focus:ring-purple-200 outline-none transition-all"
                  />
                </div>
              </div>
              <span className="text-slate-400 select-none">{')'}</span>
            </div>
          </div>

          {/* Composite Matrix C */}
          <div className="bg-slate-100 p-4 rounded-xl border border-slate-200">
            <h2 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">
              Composite Matrix C = BA
            </h2>
            <div className="flex items-center justify-center gap-2 font-mono text-slate-700">
              <span>(</span>
              <div className="grid grid-cols-2 gap-x-4 text-lg">
                <div className="text-right">{matrixC.a.toFixed(2)}</div>
                <div className="text-right">{matrixC.b.toFixed(2)}</div>
                <div className="text-right">{matrixC.c.toFixed(2)}</div>
                <div className="text-right">{matrixC.d.toFixed(2)}</div>
              </div>
              <span>)</span>
            </div>
          </div>

          {/* Preset buttons */}
          <div className="grid grid-cols-2 gap-2">
            <button
              onClick={playAnimation}
              className="col-span-2 flex items-center justify-center gap-2 px-4 py-3 bg-blue-600 text-white rounded-lg text-sm font-bold hover:bg-blue-700 transition-colors shadow-sm mb-2"
            >
              <Play size={18} fill="currentColor" /> Play Transformation (Identity → A → BA)
            </button>

            <button
              onClick={playSVD}
              className="col-span-2 flex items-center justify-center gap-2 px-4 py-3 bg-emerald-600 text-white rounded-lg text-sm font-bold hover:bg-emerald-700 transition-colors shadow-sm mb-2"
            >
              <Play size={18} fill="currentColor" /> Play SVD (Rotation → Scaling → Rotation)
            </button>

            {/* Inverse Button */}
            <button
              onClick={handleApplyInverse}
              disabled={isSingularA}
              className={`col-span-2 flex items-center justify-center gap-2 px-4 py-3 rounded-lg text-sm font-bold transition-colors shadow-sm mb-2 ${
                isSingularA
                  ? "bg-slate-100 text-slate-400 cursor-not-allowed border border-slate-200"
                  : "bg-purple-600 text-white hover:bg-purple-700"
              }`}
            >
              {isSingularA ? "No Inverse (det = 0)" : "Apply Inverse (B = A⁻¹)"}
            </button>

            <button
              onClick={() => applyPreset('identity')}
              className="flex items-center justify-center gap-2 px-4 py-2 bg-white border border-slate-200 rounded-lg text-sm font-medium text-slate-600 hover:bg-slate-50 hover:text-blue-600 transition-colors"
            >
              <RefreshCcw size={16} /> Reset A
            </button>
            <button
              onClick={() => applyPreset('rotate90')}
              className="flex items-center justify-center gap-2 px-4 py-2 bg-white border border-slate-200 rounded-lg text-sm font-medium text-slate-600 hover:bg-slate-50 hover:text-blue-600 transition-colors"
            >
              <RefreshCcw size={16} className="rotate-90" /> A: Rotate 90°
            </button>
            <button
              onClick={() => applyPreset('scale2')}
              className="flex items-center justify-center gap-2 px-4 py-2 bg-white border border-slate-200 rounded-lg text-sm font-medium text-slate-600 hover:bg-slate-50 hover:text-blue-600 transition-colors"
            >
              <Maximize2 size={16} /> A: Scale 2x
            </button>
            <button
              onClick={() => applyPreset('shearX')}
              className="flex items-center justify-center gap-2 px-4 py-2 bg-white border border-slate-200 rounded-lg text-sm font-medium text-slate-600 hover:bg-slate-50 hover:text-blue-600 transition-colors"
            >
              <MoveDiagonal size={16} /> A: Shear (X)
            </button>
          </div>

          {/* Explanation panel */}
          <div className="bg-blue-50 border border-blue-100 rounded-xl p-5">
            <div className="flex items-start gap-3">
              <Info className="text-blue-500 mt-0.5 shrink-0" size={20} />
              <div>
                <div className="flex items-center gap-2 mb-1">
                  {svdStep > 0 ? (
                    <span className="text-xs font-bold uppercase px-2 py-0.5 rounded bg-emerald-200 text-emerald-800">
                      SVD: Step {svdStep} ({svdStep === 1 ? "Rotation Vᵀ" : svdStep === 2 ? "Scaling Σ" : "Rotation U"})
                    </span>
                  ) : (
                    <span className="text-xs font-bold uppercase px-2 py-0.5 rounded bg-blue-200 text-blue-800">
                      {animationStep === 'idle' ? 'Initial State' : animationStep === 'stepA' ? 'Transformation A' : 'Composite BA'}
                    </span>
                  )}
                  <h3 className="font-bold text-blue-900">{analysis.title}</h3>
                </div>
                <p className="text-sm text-blue-800 leading-relaxed">
                  {analysis.description}
                </p>
                <div className="mt-3 pt-3 border-t border-blue-200 text-xs text-blue-700 font-mono">
                  det = {det.toFixed(3)}
                </div>
                
                {/* Display eigenvalue information */}
                <div className="mt-2 text-xs text-blue-700 font-mono">
                  {eigenData ? (
                    <>
                      <div>λ₁ = {eigenData.values[0].toFixed(2)}</div>
                      <div>λ₂ = {eigenData.values[1].toFixed(2)}</div>
                    </>
                  ) : (
                    <div className="text-slate-500 italic">
                      Eigenvalues are complex
                      <br />(Contains rotation component)
                    </div>
                  )}
                </div>

                {/* Display eigendecomposition */}
                {diagonalization && diagonalization.Pinv && (
                  <div className="mt-4 pt-4 border-t border-blue-200">
                    <h4 className="text-xs font-bold text-blue-900 uppercase mb-2">Eigendecomposition (Diagonalization)</h4>
                    <p className="text-xs text-blue-800 mb-3 leading-relaxed">
                      This matrix can be decomposed into simple scaling by <span className="font-mono">λ₁, λ₂</span> along the eigenvector directions.
                    </p>
                    
                    <div className="flex flex-wrap items-center justify-center gap-1 overflow-x-auto pb-2">
                      <MatrixTex m={currentMatrix} />
                      <span className="text-blue-800 font-bold mx-1">=</span>
                      <MatrixTex m={diagonalization.P} />
                      <MatrixTex m={diagonalization.D} />
                      <MatrixTex m={diagonalization.Pinv} />
                    </div>
                    <div className="flex justify-center gap-10 text-[10px] text-blue-500 font-mono">
                      <span className="w-16 text-center">A</span>
                      <span className="w-16 text-center">P</span>
                      <span className="w-16 text-center">D</span>
                      <span className="w-16 text-center">P⁻¹</span>
                    </div>
                  </div>
                )}

                {/* Display SVD decomposition */}
                <div className="mt-4 pt-4 border-t border-blue-200">
                  <h4 className="text-xs font-bold text-emerald-800 uppercase mb-2">Singular Value Decomposition (SVD)</h4>
                  <p className="text-xs text-emerald-900 mb-3 leading-relaxed">
                    Any matrix can be represented in 3 steps: "Rotation (Vᵀ) → Scaling (Σ) → Rotation (U)".
                    <br/>
                    Singular Values: σ₁={svdData.s1.toFixed(2)}, σ₂={svdData.s2.toFixed(2)}
                  </p>
                  
                  <div className="flex flex-wrap items-center justify-center gap-1 overflow-x-auto pb-2">
                    <MatrixTex m={matrixA} />
                    <span className="text-emerald-800 font-bold mx-1">=</span>
                    <MatrixTex m={svdData.U} />
                    <MatrixTex m={svdData.Sigma} />
                    <MatrixTex m={svdData.Vt} />
                  </div>
                  <div className="flex justify-center gap-10 text-[10px] text-emerald-600 font-mono">
                    <span className="w-16 text-center">A</span>
                    <span className="w-16 text-center">U (Rotation)</span>
                    <span className="w-16 text-center">Σ (Scaling)</span>
                    <span className="w-16 text-center">Vᵀ (Rotation)</span>
                  </div>
                </div>

              </div>
            </div>
          </div>
        </div>

        {/* Right column: Visualization */}
        <div className="lg:col-span-2 bg-white rounded-2xl shadow-lg border border-slate-200 overflow-hidden relative min-h-[400px] flex items-center justify-center">
          <div className="absolute top-4 left-4 z-10 bg-white/80 backdrop-blur px-3 py-1.5 rounded-md text-xs font-mono text-slate-500 border border-slate-200">
            Grid: 1 unit
          </div>
          
          {/* SVG Canvas */}
          <svg
            viewBox={`${-VIEWBOX_SIZE} ${-VIEWBOX_SIZE} ${VIEWBOX_SIZE * 2} ${VIEWBOX_SIZE * 2}`}
            className="w-full h-full max-h-[600px] touch-none select-none"
            preserveAspectRatio="xMidYMid meet"
          >
            {/* Background fixed grid (light gray) */}
            <g className="opacity-20">
              {gridLines.map((line) => {
                const start = toSvg(line.start.x, line.start.y);
                const end = toSvg(line.end.x, line.end.y);
                return (
                  <line
                    key={`bg-${line.key}`}
                    x1={start.x}
                    y1={start.y}
                    x2={end.x}
                    y2={end.y}
                    stroke={line.isAxis ? "#000" : "#94a3b8"}
                    strokeWidth={line.isAxis ? 0.1 : 0.05}
                  />
                );
              })}
            </g>

            {/* Transformed grid (animation) */}
            <g>
              {gridLines.map((line) => {
                // Transform start and end points
                const tStart = transformVector(currentMatrix, line.start);
                const tEnd = transformVector(currentMatrix, line.end);
                
                // To SVG coordinate system
                const svgStart = toSvg(tStart.x, tStart.y);
                const svgEnd = toSvg(tEnd.x, tEnd.y);

                return (
                  <motion.line
                    key={`fg-${line.key}`}
                    initial={false}
                    animate={{
                      x1: svgStart.x,
                      y1: svgStart.y,
                      x2: svgEnd.x,
                      y2: svgEnd.y,
                    }}
                    transition={{ type: "spring", stiffness: 200, damping: 25 }}
                    stroke={line.isAxis ? "#334155" : "#3b82f6"}
                    strokeWidth={line.isAxis ? 0.15 : 0.08}
                    strokeOpacity={line.isAxis ? 1 : 0.4}
                  />
                );
              })}
            </g>

            {/* Visualize determinant (area): Parallelogram */}
            <g>
              <motion.path
                initial={false}
                animate={{
                  d: `M ${originSvg.x} ${originSvg.y} L ${iSvg.x} ${iSvg.y} L ${sumSvg.x} ${sumSvg.y} L ${jSvg.x} ${jSvg.y} Z`,
                  fill: det >= 0 ? "#3b82f6" : "#ef4444",
                  stroke: det >= 0 ? "#3b82f6" : "#ef4444",
                }}
                transition={{ type: "spring", stiffness: 200, damping: 25 }}
                fillOpacity={0.2}
                strokeWidth={0.05}
                strokeDasharray="0.1 0.1"
              />
              <motion.text
                initial={false}
                animate={{
                  x: centerSvg.x,
                  y: centerSvg.y,
                  fill: det >= 0 ? "#1e3a8a" : "#7f1d1d",
                }}
                transition={{ type: "spring", stiffness: 200, damping: 25 }}
                fontSize="0.35"
                fontWeight="bold"
                textAnchor="middle"
                dominantBaseline="middle"
                style={{ pointerEvents: 'none', textShadow: '0px 0px 2px rgba(255,255,255,0.8)' }}
              >
                Area: {Math.abs(det).toFixed(2)}
              </motion.text>
            </g>

            {/* Visualize Unit Circle -> Ellipse */}
            <g>
              {/* Original unit circle (dotted) */}
              <path
                d={unitCirclePath}
                fill="none"
                stroke="#94a3b8"
                strokeWidth="0.04"
                strokeDasharray="0.1 0.1"
                opacity="0.6"
              />
              {/* Transformed ellipse */}
              <motion.path
                initial={false}
                animate={{ d: transformedEllipsePath }}
                transition={{ type: "spring", stiffness: 200, damping: 25 }}
                fill="rgba(139, 92, 246, 0.05)" // Light purple fill
                stroke="#8b5cf6" // Violet
                strokeWidth="0.08"
              />
            </g>

            {/* Point Grid */}
            <g>
              {dots.map((dot) => {
                const tDot = transformVector(currentMatrix, dot);
                const svgPos = toSvg(tDot.x, tDot.y);
                const isI = dot.x === 1 && dot.y === 0;
                const isJ = dot.x === 0 && dot.y === 1;

                return (
                  <motion.circle
                    key={`dot-${dot.x}-${dot.y}`}
                    initial={false}
                    animate={{ cx: svgPos.x, cy: svgPos.y }}
                    transition={{ type: "spring", stiffness: 200, damping: 25 }}
                    r={isI || isJ ? 0.15 : 0.06}
                    fill={isI ? "#fca5a5" : isJ ? "#86efac" : "#64748b"}
                    opacity={isI || isJ ? 1 : 0.4}
                  />
                );
              })}
            </g>

            {/* Eigenvectors */}
            {eigenData && (
              <g>
                {eigenData.vectors.map((v, idx) => {
                  // Do not draw if zero vector
                  if (Math.abs(v.x) < 1e-6 && Math.abs(v.y) < 1e-6) return null;

                  const color = idx === 0 ? "#eab308" : "#a855f7"; // Yellow / Purple
                  const svgV = toSvg(v.x, v.y);
                  
                  // Coordinates for guidelines (extend to screen edge)
                  const scale = VIEWBOX_SIZE * 2;
                  const guideStart = toSvg(-v.x * scale, -v.y * scale);
                  const guideEnd = toSvg(v.x * scale, v.y * scale);

                  return (
                    <React.Fragment key={`eigen-${idx}`}>
                      {/* Infinite guidelines (dotted) */}
                      <motion.line
                        initial={false}
                        animate={{
                          x1: guideStart.x, y1: guideStart.y,
                          x2: guideEnd.x, y2: guideEnd.y
                        }}
                        transition={{ type: "spring", stiffness: 200, damping: 25 }}
                        stroke={color}
                        strokeWidth={0.05}
                        strokeDasharray="0.2 0.2"
                        opacity={0.6}
                      />
                      {/* Eigenvector body (arrow) */}
                      <VectorArrow
                        vector={v}
                        color={color}
                        label={`v${idx + 1}`}
                        isHovered={false}
                        onHover={() => {}}
                      />
                    </React.Fragment>
                  );
                })}
              </g>
            )}

            {/* Basis vector i (red) */}
            <VectorArrow
              vector={transformedI}
              color="#ef4444"
              label="i"
              isHovered={hoveredVector === 'i'}
              onHover={(v) => setHoveredVector(v ? 'i' : null)}
            />

            {/* Basis vector j (green) */}
            <VectorArrow
              vector={transformedJ}
              color="#22c55e"
              label="j"
              isHovered={hoveredVector === 'j'}
              onHover={(v) => setHoveredVector(v ? 'j' : null)}
            />

            {/* Origin dot */}
            <circle cx={0} cy={0} r={0.2} fill="#1e293b" />
          </svg>
        </div>
      </div>
    </div>
  );
}

// --- Subcomponent: Vector Arrow ---
function VectorArrow({
  vector,
  color,
  label,
  isHovered,
  onHover
}: {
  vector: Vector;
  color: string;
  label: string;
  isHovered: boolean;
  onHover: (hover: boolean) => void;
}) {
  const svgEnd = toSvg(vector.x, vector.y);
  const length = Math.sqrt(vector.x ** 2 + vector.y ** 2);
  
  // Adjust arrow head size (smaller if vector is too short)
  const headSize = Math.min(0.5, length * 0.4); 
  
  // Calculate arrow angle
  const angle = Math.atan2(-vector.y, vector.x); // Angle in SVG coordinate system

  return (
    <motion.g
      initial={false}
      animate={{ x: 0, y: 0 }} // Group position fixed, internal coordinates move
      onMouseEnter={() => onHover(true)}
      onMouseLeave={() => onHover(false)}
      className="cursor-pointer"
    >
      {/* Main line */}
      <motion.line
        x1={0}
        y1={0}
        animate={{ x2: svgEnd.x, y2: svgEnd.y }}
        transition={{ type: "spring", stiffness: 200, damping: 25 }}
        stroke={color}
        strokeWidth={isHovered ? 0.2 : 0.12}
        strokeLinecap="round"
      />

      {/* Arrow tip */}
      <motion.path
        d={`M -${headSize} -${headSize/2} L 0 0 L -${headSize} ${headSize/2}`}
        animate={{
          translateX: svgEnd.x,
          translateY: svgEnd.y,
          rotate: angle * (180 / Math.PI),
        }}
        transition={{ type: "spring", stiffness: 200, damping: 25 }}
        fill="none"
        stroke={color}
        strokeWidth={isHovered ? 0.2 : 0.12}
        strokeLinecap="round"
        strokeLinejoin="round"
      />

      {/* Label */}
      <motion.text
        animate={{
          x: svgEnd.x + (vector.x >= 0 ? 0.3 : -0.8),
          y: svgEnd.y + (vector.y >= 0 ? -0.3 : 0.8),
        }}
        transition={{ type: "spring", stiffness: 200, damping: 25 }}
        fill={color}
        fontSize="0.8"
        fontWeight="bold"
        style={{ pointerEvents: 'none' }}
      >
        {label}
        {isHovered && (
          <tspan fontSize="0.5" dx="0.2" fill="#64748b">
            ({vector.x.toFixed(1)}, {vector.y.toFixed(1)})
          </tspan>
        )}
      </motion.text>
    </motion.g>
  );
}

// --- Subcomponent: Matrix Display for Math ---
function MatrixTex({ m }: { m: Matrix }) {
  return (
    <div className="inline-flex items-center">
      <div className="relative px-1.5 py-1 border-l border-r border-slate-800 rounded-[4px]">
        <div className="grid grid-cols-2 gap-x-2 gap-y-0.5 text-center font-mono text-[11px] leading-tight text-slate-800">
          <div className="text-right min-w-[24px]">{m.a.toFixed(2)}</div>
          <div className="text-right min-w-[24px]">{m.b.toFixed(2)}</div>
          <div className="text-right min-w-[24px]">{m.c.toFixed(2)}</div>
          <div className="text-right min-w-[24px]">{m.d.toFixed(2)}</div>
        </div>
      </div>
    </div>
  );
}
