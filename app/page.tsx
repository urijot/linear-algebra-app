'use client';

import React, { useState, useEffect, useMemo } from 'react';
import { motion } from 'framer-motion';
import { RefreshCcw, Info, MousePointer2, Maximize2, MoveDiagonal, Play, ArrowRight } from 'lucide-react';

// --- 型定義 ---
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

// --- 定数 ---
const GRID_SIZE = 10; // グリッドの範囲 (-10 ~ 10)
const GRID_STEP = 1;  // グリッドの間隔
const VIEWBOX_SIZE = 12; // 表示範囲

// --- ヘルパー関数 ---

// 数学座標(x, y)をSVG座標(x, y)に変換（Y軸を反転）
const toSvg = (x: number, y: number) => ({ x: x, y: -y });

// 行列とベクトルの積を計算
const transformVector = (matrix: Matrix, vector: Vector): Vector => {
  return {
    x: matrix.a * vector.x + matrix.b * vector.y,
    y: matrix.c * vector.x + matrix.d * vector.y,
  };
};

// 行列の積を計算 (C = B * A)
const multiplyMatrices = (m1: Matrix, m2: Matrix): Matrix => {
  return {
    a: m1.a * m2.a + m1.b * m2.c,
    b: m1.a * m2.b + m1.b * m2.d,
    c: m1.c * m2.a + m1.d * m2.c,
    d: m1.c * m2.b + m1.d * m2.d,
  };
};

// 逆行列を計算 (Inverse Matrix)
const invertMatrix = (m: Matrix): Matrix | null => {
  const det = m.a * m.d - m.b * m.c;
  // 行列式が0に近い場合は逆行列が存在しない
  if (Math.abs(det) < 1e-6) return null;
  const invDet = 1 / det;
  return {
    a: m.d * invDet,
    b: -m.b * invDet,
    c: -m.c * invDet,
    d: m.a * invDet,
  };
};

// 行列の性質を分析して解説テキストを返す
const analyzeMatrix = (m: Matrix): { title: string; description: string } => {
  const det = m.a * m.d - m.b * m.c;
  const epsilon = 0.001;
  const isClose = (v1: number, v2: number) => Math.abs(v1 - v2) < epsilon;

  if (isClose(det, 0)) {
    return {
      title: '特異行列 (Singular)',
      description: '行列式が0です。空間が直線または点に潰れてしまい、逆変換ができません。',
    };
  }

  if (isClose(m.a, 1) && isClose(m.b, 0) && isClose(m.c, 0) && isClose(m.d, 1)) {
    return {
      title: '単位行列 (Identity)',
      description: '何も変化させない変換です。',
    };
  }

  // 回転行列の判定: [[cos, -sin], [sin, cos]]
  // a = d, b = -c, a^2 + c^2 = 1 (スケーリングなしの場合)
  // ここでは回転成分が含まれているかを簡易判定
  if (isClose(m.a, m.d) && isClose(m.b, -m.c) && !isClose(m.b, 0)) {
    const angle = Math.atan2(m.c, m.a) * (180 / Math.PI);
    return {
      title: '回転 (Rotation)',
      description: `原点を中心に約 ${Math.round(angle)}度 回転させる変換です（拡大縮小を含む場合があります）。`,
    };
  }

  // 拡大縮小: b=0, c=0
  if (isClose(m.b, 0) && isClose(m.c, 0)) {
    if (isClose(m.a, m.d)) {
      return {
        title: '一様拡大・縮小 (Uniform Scaling)',
        description: `全体を ${m.a.toFixed(2)}倍 に拡大・縮小します。`,
      };
    }
    return {
      title: '非一様拡大・縮小 (Non-uniform Scaling)',
      description: `X軸方向に ${m.a}倍、Y軸方向に ${m.d}倍 します。`,
    };
  }

  // 剪断 (Shear)
  // X軸剪断: a=1, c=0, d=1, b!=0
  if (isClose(m.a, 1) && isClose(m.c, 0) && isClose(m.d, 1) && !isClose(m.b, 0)) {
    return {
      title: '剪断 (Shear X)',
      description: 'X軸方向に平行にズラす変換です（Y座標に依存してXが変化）。',
    };
  }
  // Y軸剪断: a=1, b=0, d=1, c!=0
  if (isClose(m.a, 1) && isClose(m.b, 0) && isClose(m.d, 1) && !isClose(m.c, 0)) {
    return {
      title: '剪断 (Shear Y)',
      description: 'Y軸方向に平行にズラす変換です（X座標に依存してYが変化）。',
    };
  }

  return {
    title: '一般的な線形変換',
    description: `行列式: ${det.toFixed(2)}。基底ベクトル i, j がそれぞれ新しい位置に移ります。`,
  };
};

// 固有値・固有ベクトルの計算
const calculateEigen = (m: Matrix): { values: number[]; vectors: Vector[] } | null => {
  const trace = m.a + m.d;
  const det = m.a * m.d - m.b * m.c;
  const discriminant = trace * trace - 4 * det;

  // 判別式が負の場合は複素数解（ここでは可視化しない）
  if (discriminant < -0.0001) return null;

  const sqrtD = Math.sqrt(Math.max(0, discriminant));
  const l1 = (trace + sqrtD) / 2;
  const l2 = (trace - sqrtD) / 2;

  // 固有ベクトルの計算 (A - lambda I)v = 0
  const getEigenVector = (lambda: number): Vector => {
    // 対角行列 (b=0, c=0) の場合の特別処理
    if (Math.abs(m.b) < 1e-6 && Math.abs(m.c) < 1e-6) {
      // lambda が a に近ければ (1,0)、d に近ければ (0,1)
      // 重解 (a=d) の場合は後続の処理で分離するが、ここでは基本軸を返す
      if (Math.abs(lambda - m.a) < 1e-6) return { x: 1, y: 0 };
      return { x: 0, y: 1 };
    }

    // 一般形:
    // b != 0 なら x=b, y=lambda-a
    if (Math.abs(m.b) > 1e-6) {
      return { x: m.b, y: lambda - m.a };
    }
    // c != 0 なら x=lambda-d, y=c
    if (Math.abs(m.c) > 1e-6) {
      return { x: lambda - m.d, y: m.c };
    }
    
    return { x: 0, y: 0 };
  };

  let v1 = getEigenVector(l1);
  let v2 = getEigenVector(l2);

  // 対角行列かつ重解の場合（単位行列など）は、直交する基底を強制的に割り当てる
  if (Math.abs(m.b) < 1e-6 && Math.abs(m.c) < 1e-6 && Math.abs(l1 - l2) < 1e-6) {
      v1 = { x: 1, y: 0 };
      v2 = { x: 0, y: 1 };
  }

  // 正規化 (長さ1にする)
  const normalize = (v: Vector) => {
    const len = Math.sqrt(v.x * v.x + v.y * v.y);
    return len < 1e-6 ? { x: 0, y: 0 } : { x: v.x / len, y: v.y / len };
  };

  return {
    values: [l1, l2],
    vectors: [normalize(v1), normalize(v2)],
  };
};

// --- コンポーネント ---

export default function LinearAlgebraPage() {
  const [matrixA, setMatrixA] = useState<Matrix>({ a: 1, b: 0, c: 0, d: 1 });
  const [matrixB, setMatrixB] = useState<Matrix>({ a: 1, b: 0, c: 0, d: 1 });
  const [animationStep, setAnimationStep] = useState<'idle' | 'stepA' | 'stepBA'>('stepBA');
  const [hoveredVector, setHoveredVector] = useState<'i' | 'j' | null>(null);

  // グリッド線の生成
  const gridLines = useMemo(() => {
    const lines = [];
    // 垂直線 (x = k)
    for (let x = -GRID_SIZE; x <= GRID_SIZE; x += GRID_STEP) {
      lines.push({
        key: `v${x}`,
        start: { x, y: -GRID_SIZE },
        end: { x, y: GRID_SIZE },
        isAxis: x === 0,
      });
    }
    // 水平線 (y = k)
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

  // ドット格子の生成
  const dots = useMemo(() => {
    const points = [];
    for (let x = -GRID_SIZE; x <= GRID_SIZE; x += GRID_STEP) {
      for (let y = -GRID_SIZE; y <= GRID_SIZE; y += GRID_STEP) {
        // 原点は別途描画されているため除外
        if (x === 0 && y === 0) continue;
        points.push({ x, y });
      }
    }
    return points;
  }, []);

  // 合成行列の計算 (C = B * A)
  const matrixC = useMemo(() => multiplyMatrices(matrixB, matrixA), [matrixA, matrixB]);

  // 現在表示すべき行列（アニメーション用）
  const currentMatrix = useMemo(() => {
    switch (animationStep) {
      case 'idle': // 初期状態（単位行列）
        return { a: 1, b: 0, c: 0, d: 1 };
      case 'stepA': // Aによる変換
        return matrixA;
      case 'stepBA': // B(A)による変換（最終結果）
        return matrixC;
      default:
        return matrixC;
    }
  }, [animationStep, matrixA, matrixC]);

  // 変換後の基底ベクトル
  const transformedI = transformVector(currentMatrix, { x: 1, y: 0 });
  const transformedJ = transformVector(currentMatrix, { x: 0, y: 1 });

  // 平行四辺形の4点目と行列式（面積）
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

  // 解析結果
  const analysis = analyzeMatrix(currentMatrix);
  
  // 固有値・固有ベクトルの計算
  const eigenData = useMemo(() => calculateEigen(currentMatrix), [currentMatrix]);

  // 固有値分解の計算 (A = PDP^-1)
  const diagonalization = useMemo(() => {
    if (!eigenData) return null;
    
    const [l1, l2] = eigenData.values;
    const [v1, v2] = eigenData.vectors;
    
    // P = [v1 v2] (固有ベクトルを列に並べる)
    const P: Matrix = {
      a: v1.x, b: v2.x,
      c: v1.y, d: v2.y
    };
    
    // D = diag(l1, l2) (固有値を対角に並べる)
    const D: Matrix = {
      a: l1, b: 0,
      c: 0, d: l2
    };
    
    // Pの逆行列
    const Pinv = invertMatrix(P);
    
    return { P, D, Pinv };
  }, [eigenData]);

  // 単位円と変換後の楕円のパス生成
  const { unitCirclePath, transformedEllipsePath } = useMemo(() => {
    const steps = 90; // 4度刻み
    let unitPath = "";
    let ellipsePath = "";

    for (let i = 0; i <= steps; i++) {
      const theta = (i / steps) * Math.PI * 2;
      const cos = Math.cos(theta);
      const sin = Math.sin(theta);
      
      // 単位円 (半径1)
      const p = toSvg(cos, sin);
      unitPath += `${i === 0 ? "M" : "L"} ${p.x} ${p.y}`;

      // 変換後の楕円
      const t = transformVector(currentMatrix, { x: cos, y: sin });
      const tp = toSvg(t.x, t.y);
      ellipsePath += `${i === 0 ? "M" : "L"} ${tp.x} ${tp.y}`;
    }

    return {
      unitCirclePath: unitPath + " Z",
      transformedEllipsePath: ellipsePath + " Z"
    };
  }, [currentMatrix]);

  // 入力ハンドラ
  const handleInputChange = (target: 'A' | 'B', key: keyof Matrix, value: string) => {
    const num = parseFloat(value);
    if (!isNaN(num)) {
      if (target === 'A') {
        setMatrixA((prev) => ({ ...prev, [key]: num }));
      } else {
        setMatrixB((prev) => ({ ...prev, [key]: num }));
      }
      // 編集中は最終結果を表示
      setAnimationStep('stepBA');
    } else if (value === '' || value === '-') {
       // 入力途中を許容するための処理（実際にはstateをstringで持つか、onBlurで確定する方がUXが良いが、プロトタイプとして簡易実装）
    }
  };

  // 逆変換の適用ハンドラ
  const detA = matrixA.a * matrixA.d - matrixA.b * matrixA.c;
  const isSingularA = Math.abs(detA) < 1e-4;

  const handleApplyInverse = () => {
    const inv = invertMatrix(matrixA);
    if (inv) {
      setMatrixB(inv);
      setAnimationStep('stepBA');
    }
  };

  // プリセット適用
  const applyPreset = (type: 'identity' | 'rotate90' | 'scale2' | 'shearX') => {
    // プリセット適用時はBをリセットし、Aに適用する
    setMatrixB({ a: 1, b: 0, c: 0, d: 1 });
    setAnimationStep('stepBA');

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

  // アニメーション再生
  const playAnimation = () => {
    setAnimationStep('idle');
    setTimeout(() => setAnimationStep('stepA'), 800);
    setTimeout(() => setAnimationStep('stepBA'), 2000);
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 p-4 md:p-8 font-sans">
      <div className="max-w-6xl mx-auto grid grid-cols-1 lg:grid-cols-3 gap-8">
        
        {/* 左カラム: コントロールパネル */}
        <div className="lg:col-span-1 space-y-6">
          <header>
            <h1 className="text-2xl font-bold text-slate-800 mb-2">線形変換ビジュアライザー</h1>
            <p className="text-slate-600 text-sm">
              行列 $A$ で変換し、さらに行列 $B$ で変換する合成変換 ($C = BA$) を観察しましょう。
            </p>
          </header>

          {/* 行列入力フォーム A */}
          <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
            <h2 className="text-sm font-semibold text-slate-500 uppercase tracking-wider mb-4 flex items-center gap-2">
              1. 変換行列 A (最初に適用)
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

          {/* 行列入力フォーム B */}
          <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
            <h2 className="text-sm font-semibold text-slate-500 uppercase tracking-wider mb-4 flex items-center gap-2">
              2. 変換行列 B (次に適用)
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

          {/* 合成行列 C の表示 */}
          <div className="bg-slate-100 p-4 rounded-xl border border-slate-200">
            <h2 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">
              合成行列 C = BA
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

          {/* プリセットボタン */}
          <div className="grid grid-cols-2 gap-2">
            <button
              onClick={playAnimation}
              className="col-span-2 flex items-center justify-center gap-2 px-4 py-3 bg-blue-600 text-white rounded-lg text-sm font-bold hover:bg-blue-700 transition-colors shadow-sm mb-2"
            >
              <Play size={18} fill="currentColor" /> 変換を再生 (Identity → A → BA)
            </button>

            {/* 逆変換ボタン */}
            <button
              onClick={handleApplyInverse}
              disabled={isSingularA}
              className={`col-span-2 flex items-center justify-center gap-2 px-4 py-3 rounded-lg text-sm font-bold transition-colors shadow-sm mb-2 ${
                isSingularA
                  ? "bg-slate-100 text-slate-400 cursor-not-allowed border border-slate-200"
                  : "bg-purple-600 text-white hover:bg-purple-700"
              }`}
            >
              {isSingularA ? "逆行列なし (det = 0)" : "逆変換を適用 (B = A⁻¹)"}
            </button>

            <button
              onClick={() => applyPreset('identity')}
              className="flex items-center justify-center gap-2 px-4 py-2 bg-white border border-slate-200 rounded-lg text-sm font-medium text-slate-600 hover:bg-slate-50 hover:text-blue-600 transition-colors"
            >
              <RefreshCcw size={16} /> Aをリセット
            </button>
            <button
              onClick={() => applyPreset('rotate90')}
              className="flex items-center justify-center gap-2 px-4 py-2 bg-white border border-slate-200 rounded-lg text-sm font-medium text-slate-600 hover:bg-slate-50 hover:text-blue-600 transition-colors"
            >
              <RefreshCcw size={16} className="rotate-90" /> A: 90° 回転
            </button>
            <button
              onClick={() => applyPreset('scale2')}
              className="flex items-center justify-center gap-2 px-4 py-2 bg-white border border-slate-200 rounded-lg text-sm font-medium text-slate-600 hover:bg-slate-50 hover:text-blue-600 transition-colors"
            >
              <Maximize2 size={16} /> A: 2倍 拡大
            </button>
            <button
              onClick={() => applyPreset('shearX')}
              className="flex items-center justify-center gap-2 px-4 py-2 bg-white border border-slate-200 rounded-lg text-sm font-medium text-slate-600 hover:bg-slate-50 hover:text-blue-600 transition-colors"
            >
              <MoveDiagonal size={16} /> A: 剪断 (X)
            </button>
          </div>

          {/* 解説パネル */}
          <div className="bg-blue-50 border border-blue-100 rounded-xl p-5">
            <div className="flex items-start gap-3">
              <Info className="text-blue-500 mt-0.5 shrink-0" size={20} />
              <div>
                <div className="flex items-center gap-2 mb-1">
                  <span className="text-xs font-bold uppercase px-2 py-0.5 rounded bg-blue-200 text-blue-800">
                    {animationStep === 'idle' ? '初期状態' : animationStep === 'stepA' ? '変換 A' : '合成変換 BA'}
                  </span>
                  <h3 className="font-bold text-blue-900">{analysis.title}</h3>
                </div>
                <p className="text-sm text-blue-800 leading-relaxed">
                  {analysis.description}
                </p>
                <div className="mt-3 pt-3 border-t border-blue-200 text-xs text-blue-700 font-mono">
                  det = {det.toFixed(3)}
                </div>
                
                {/* 固有値情報の表示 */}
                <div className="mt-2 text-xs text-blue-700 font-mono">
                  {eigenData ? (
                    <>
                      <div>λ₁ = {eigenData.values[0].toFixed(2)}</div>
                      <div>λ₂ = {eigenData.values[1].toFixed(2)}</div>
                    </>
                  ) : (
                    <div className="text-slate-500 italic">
                      固有値は複素数です
                      <br />(回転成分が含まれています)
                    </div>
                  )}
                </div>

                {/* 固有値分解の表示 */}
                {diagonalization && diagonalization.Pinv && (
                  <div className="mt-4 pt-4 border-t border-blue-200">
                    <h4 className="text-xs font-bold text-blue-900 uppercase mb-2">固有値分解 (Diagonalization)</h4>
                    <p className="text-xs text-blue-800 mb-3 leading-relaxed">
                      この行列は、固有ベクトルの方向に <span className="font-mono">λ₁, λ₂</span> 倍するだけの単純な動きに分解できます。
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
              </div>
            </div>
          </div>
        </div>

        {/* 右カラム: ビジュアライゼーション */}
        <div className="lg:col-span-2 bg-white rounded-2xl shadow-lg border border-slate-200 overflow-hidden relative min-h-[400px] flex items-center justify-center">
          <div className="absolute top-4 left-4 z-10 bg-white/80 backdrop-blur px-3 py-1.5 rounded-md text-xs font-mono text-slate-500 border border-slate-200">
            Grid: 1 unit
          </div>
          
          {/* SVG キャンバス */}
          <svg
            viewBox={`${-VIEWBOX_SIZE} ${-VIEWBOX_SIZE} ${VIEWBOX_SIZE * 2} ${VIEWBOX_SIZE * 2}`}
            className="w-full h-full max-h-[600px] touch-none select-none"
            preserveAspectRatio="xMidYMid meet"
          >
            {/* 背景の固定グリッド (薄いグレー) */}
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

            {/* 変換されるグリッド (アニメーション) */}
            <g>
              {gridLines.map((line) => {
                // 始点と終点を変換
                const tStart = transformVector(currentMatrix, line.start);
                const tEnd = transformVector(currentMatrix, line.end);
                
                // SVG座標系へ
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

            {/* 行列式（面積）の可視化: 平行四辺形 */}
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

            {/* 単位円 -> 楕円の可視化 */}
            <g>
              {/* 元の単位円 (点線) */}
              <path
                d={unitCirclePath}
                fill="none"
                stroke="#94a3b8"
                strokeWidth="0.04"
                strokeDasharray="0.1 0.1"
                opacity="0.6"
              />
              {/* 変換後の楕円 */}
              <motion.path
                initial={false}
                animate={{ d: transformedEllipsePath }}
                transition={{ type: "spring", stiffness: 200, damping: 25 }}
                fill="rgba(139, 92, 246, 0.05)" // 薄い紫の塗りつぶし
                stroke="#8b5cf6" // Violet
                strokeWidth="0.08"
              />
            </g>

            {/* ドット格子 (Point Grid) */}
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

            {/* 固有ベクトル (Eigenvectors) */}
            {eigenData && (
              <g>
                {eigenData.vectors.map((v, idx) => {
                  // ゼロベクトルの場合は描画しない
                  if (Math.abs(v.x) < 1e-6 && Math.abs(v.y) < 1e-6) return null;

                  const color = idx === 0 ? "#eab308" : "#a855f7"; // 黄色 / 紫
                  const svgV = toSvg(v.x, v.y);
                  
                  // ガイドライン用の座標（画面端まで伸ばす）
                  const scale = VIEWBOX_SIZE * 2;
                  const guideStart = toSvg(-v.x * scale, -v.y * scale);
                  const guideEnd = toSvg(v.x * scale, v.y * scale);

                  return (
                    <React.Fragment key={`eigen-${idx}`}>
                      {/* 無限に続くガイドライン (点線) */}
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
                      {/* 固有ベクトル本体 (矢印) */}
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

            {/* 基底ベクトル i (赤) */}
            <VectorArrow
              vector={transformedI}
              color="#ef4444"
              label="i"
              isHovered={hoveredVector === 'i'}
              onHover={(v) => setHoveredVector(v ? 'i' : null)}
            />

            {/* 基底ベクトル j (緑) */}
            <VectorArrow
              vector={transformedJ}
              color="#22c55e"
              label="j"
              isHovered={hoveredVector === 'j'}
              onHover={(v) => setHoveredVector(v ? 'j' : null)}
            />

            {/* 原点ドット */}
            <circle cx={0} cy={0} r={0.2} fill="#1e293b" />
          </svg>
        </div>
      </div>
    </div>
  );
}

// --- サブコンポーネント: ベクトル矢印 ---
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
  
  // 矢印のヘッドサイズ調整（ベクトルが短すぎるときは小さくする）
  const headSize = Math.min(0.5, length * 0.4); 
  
  // 矢印の角度計算
  const angle = Math.atan2(-vector.y, vector.x); // SVG座標系での角度

  return (
    <motion.g
      initial={false}
      animate={{ x: 0, y: 0 }} // グループ全体の位置は固定、内部座標を動かす
      onMouseEnter={() => onHover(true)}
      onMouseLeave={() => onHover(false)}
      className="cursor-pointer"
    >
      {/* メインの線 */}
      <motion.line
        x1={0}
        y1={0}
        animate={{ x2: svgEnd.x, y2: svgEnd.y }}
        transition={{ type: "spring", stiffness: 200, damping: 25 }}
        stroke={color}
        strokeWidth={isHovered ? 0.2 : 0.12}
        strokeLinecap="round"
      />

      {/* 矢印の先端 */}
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

      {/* ラベル */}
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

// --- サブコンポーネント: 数式用行列表示 ---
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
