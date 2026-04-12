import React, { useState, useEffect, useRef, useCallback } from 'react';
import { api } from './api';
import { TreePine, Save, Layers, Download, Crop, Play, X, RotateCcw, ZoomIn, Eye, CheckCircle2, AlertCircle } from 'lucide-react';

export interface DetectionPoint { x: number; y: number; label: string; conf?: number; }
export interface ImageInfo { name: string; path: string; width: number; height: number; }

const App: React.FC = () => {
  const [images, setImages] = useState<any[]>([]);
  const [selectedImage, setSelectedImage] = useState<ImageInfo | null>(null);
  const [points, setPoints] = useState<DetectionPoint[]>([]);
  const [loading, setLoading] = useState<string | null>(null);
  const [detecting, setDetecting] = useState(false);
  const [statusMsg, setStatusMsg] = useState('');
  const [scanningBox, setScanningBox] = useState<number[] | null>(null);
  const [progress, setProgress] = useState(0);
  const abortControllerRef = useRef<AbortController | null>(null);

  const [selectionMode, setSelectionMode] = useState(false);
  const [polygon, setPolygon] = useState<[number, number][]>([]);
  const [mousePos, setMousePos] = useState<[number, number] | null>(null);
  const [roiImage, setRoiImage] = useState<{el: HTMLImageElement, x: number, y: number, w: number, h: number} | null>(null);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [scale, setScale] = useState(1);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const [imgElement, setImgElement] = useState<HTMLImageElement | null>(null);
  const isDragging = useRef(false);
  const lastMousePos = useRef({ x: 0, y: 0 });

  useEffect(() => { loadImages(); }, []);
  const loadImages = async () => { try { const resp = await api.getImages(); setImages(resp.data); } catch (err) { } };

  const handleSelectImage = async (img: any) => {
    setLoading('Loading Preview...'); setPoints([]); setImgElement(null); setScanningBox(null); setPolygon([]); setRoiImage(null); setProgress(0);
    try {
      const infoResp = await api.getImageInfo(img.path);
      const info = infoResp.data;
      setSelectedImage({ ...img, width: info.width, height: info.height });
      const previewScale = Math.min(1.0, 2500 / Math.max(info.width, info.height));
      const tileUrl = api.getTileUrl(img.path, 0, 0, info.width, info.height, previewScale);
      const imgEl = new Image();
      imgEl.src = tileUrl;
      imgEl.onload = () => {
        setImgElement(imgEl);
        const canvas = canvasRef.current;
        if (canvas) {
          const s = Math.min(canvas.clientWidth / info.width, canvas.clientHeight / info.height) * 0.9;
          setScale(s); setOffset({ x: (canvas.clientWidth - info.width * s) / 2, y: (canvas.clientHeight - info.height * s) / 2 });
        }
        setLoading(null);
      };
      const labelsResp = await api.loadLabels(img.path);
      if (labelsResp.data?.length) setPoints(labelsResp.data);
    } catch (err) { setStatusMsg('Load error'); setLoading(null); }
  };

  const drawCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx || !selectedImage) return;
    const dpr = window.devicePixelRatio || 1;
    if (canvas.width !== canvas.clientWidth * dpr || canvas.height !== canvas.clientHeight * dpr) {
      canvas.width = canvas.clientWidth * dpr; canvas.height = canvas.clientHeight * dpr;
    }
    ctx.scale(dpr, dpr);
    ctx.imageSmoothingEnabled = scale < 2.5;
    ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
    ctx.fillStyle = '#0a0a0a'; ctx.fillRect(0, 0, canvas.clientWidth, canvas.clientHeight);
    ctx.save();
    ctx.translate(offset.x, offset.y);
    ctx.scale(scale, scale);

    if (imgElement) ctx.drawImage(imgElement, 0, 0, selectedImage.width, selectedImage.height);
    if (roiImage) ctx.drawImage(roiImage.el, roiImage.x, roiImage.y, roiImage.w, roiImage.h);

    if (detecting && scanningBox) {
      ctx.strokeStyle = '#3b82f6'; ctx.lineWidth = 3 / scale;
      ctx.strokeRect(scanningBox[0], scanningBox[1], scanningBox[2], scanningBox[3]);
      ctx.fillStyle = 'rgba(59, 130, 246, 0.15)'; ctx.fillRect(scanningBox[0], scanningBox[1], scanningBox[2], scanningBox[3]);
    }

    if (polygon.length > 0) {
      ctx.strokeStyle = '#ef4444'; ctx.lineWidth = 2 / scale; ctx.fillStyle = 'rgba(239, 68, 68, 0.15)';
      ctx.beginPath();
      ctx.moveTo(polygon[0][0], polygon[0][1]);
      for (let i = 1; i < polygon.length; i++) {
        ctx.lineTo(polygon[i][0], polygon[i][1]);
      }
      if (selectionMode && mousePos) {
        ctx.lineTo(mousePos[0], mousePos[1]);
      }
      if (!selectionMode && polygon.length > 2) ctx.closePath();
      ctx.stroke(); 
      if (!selectionMode && polygon.length > 2) ctx.fill();
      
      polygon.forEach(p => {
        ctx.fillStyle = '#ef4444'; ctx.beginPath(); ctx.arc(p[0], p[1], 4/scale, 0, Math.PI*2); ctx.fill();
      });
    }

    if (points.length > 0) {
      const radius = Math.max(1.2, 3.0 / scale);
      ctx.fillStyle = '#22c55e'; ctx.strokeStyle = '#fff'; ctx.lineWidth = 0.8 / scale;
      points.forEach(p => { ctx.beginPath(); ctx.arc(p.x, p.y, radius, 0, Math.PI * 2); ctx.fill(); ctx.stroke(); });
    }
    ctx.restore();
  }, [selectedImage, imgElement, points, scale, offset, detecting, scanningBox, polygon, selectionMode, mousePos, roiImage]);

  useEffect(() => { drawCanvas(); }, [drawCanvas]);

  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const canvas = canvasRef.current; if (!canvas) return;
    const factor = e.deltaY > 0 ? 0.85 : 1.15;
    const newScale = Math.max(0.0001, Math.min(200, scale * factor));
    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left; const mouseY = e.clientY - rect.top;
    const worldX = (mouseX - offset.x) / scale; const worldY = (mouseY - offset.y) / scale;
    setOffset({ x: mouseX - worldX * newScale, y: mouseY - worldY * newScale }); setScale(newScale);
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    const canvas = canvasRef.current; if (!canvas || !selectedImage) return;
    if (selectionMode && e.button === 0) {
      const rect = canvas.getBoundingClientRect();
      const worldX = (e.clientX - rect.left - offset.x) / scale;
      const worldY = (e.clientY - rect.top - offset.y) / scale;
      setPolygon(prev => [...prev, [worldX, worldY]]); return;
    }
    // Pan: middle click OR alt+left click OR (left click when ROI is loaded and not clicking on a point)
    if (e.button === 1 || (e.button === 0 && e.altKey) || (e.button === 0 && roiImage)) {
      isDragging.current = true; lastMousePos.current = { x: e.clientX, y: e.clientY };
      e.preventDefault();
    }
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isDragging.current) {
      const dx = e.clientX - lastMousePos.current.x; const dy = e.clientY - lastMousePos.current.y;
      setOffset(prev => ({ x: prev.x + dx, y: prev.y + dy }));
      lastMousePos.current = { x: e.clientX, y: e.clientY };
    }
    if (selectionMode && canvasRef.current) {
      const rect = canvasRef.current.getBoundingClientRect();
      const worldX = (e.clientX - rect.left - offset.x) / scale;
      const worldY = (e.clientY - rect.top - offset.y) / scale;
      setMousePos([worldX, worldY]);
    }
  };

  const handleMouseUp = () => { isDragging.current = false; };

  const handleCanvasClick = (e: React.MouseEvent) => {
    if (selectionMode || isDragging.current) return;
    const canvas = canvasRef.current; if (!canvas || !selectedImage || isDragging.current) return;
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left - offset.x) / scale;
    const y = (e.clientY - rect.top - offset.y) / scale;
    if (e.button === 2) {
      const idx = points.findIndex(p => Math.hypot(p.x - x, p.y - y) < 15 / scale);
      if (idx !== -1) setPoints(prev => prev.filter((_, i) => i !== idx));
    } else if (e.button === 0 && !e.altKey) {
      setPoints(prev => [...prev, { x: Math.round(x), y: Math.round(y), label: 'palm_tree' }]);
    }
  };

  const handleCropToROI = async () => {
    if (polygon.length < 3 || !canvasRef.current || !selectedImage) return;
    const xs = polygon.map(p => p[0]), ys = polygon.map(p => p[1]);
    const minX = Math.floor(Math.max(0, Math.min(...xs)));
    const maxX = Math.ceil(Math.min(selectedImage.width, Math.max(...xs)));
    const minY = Math.floor(Math.max(0, Math.min(...ys)));
    const maxY = Math.ceil(Math.min(selectedImage.height, Math.max(...ys)));
    const rw = maxX - minX, rh = maxY - minY;

    setLoading('Fetching high-resolution crop...');
    const url = api.getTileUrl(selectedImage.path, minX, minY, rw, rh, 1.0);
    const imgEl = new Image();
    imgEl.src = url;
    imgEl.onload = () => {
      setRoiImage({ el: imgEl, x: minX, y: minY, w: rw, h: rh });
      const canvas = canvasRef.current!;
      const padding = 60;
      const newScale = Math.min((canvas.clientWidth - padding * 2) / rw, (canvas.clientHeight - padding * 2) / rh);
      setScale(newScale);
      setOffset({ x: (canvas.clientWidth - rw * newScale) / 2 - minX * newScale, y: (canvas.clientHeight - rh * newScale) / 2 - minY * newScale });
      setSelectionMode(false); setLoading(null); setStatusMsg('High-res ROI loaded.');
    };
    imgEl.onerror = () => { setLoading(null); setStatusMsg('Crop load failed'); };
  };

  const handleDetectVisibleArea = () => {
    if (!canvasRef.current || !selectedImage) return;
    const canvas = canvasRef.current;
    const x = Math.max(0, -offset.x / scale);
    const y = Math.max(0, -offset.y / scale);
    const w = Math.min(selectedImage.width, canvas.clientWidth / scale);
    const h = Math.min(selectedImage.height, canvas.clientHeight / scale);
    setPolygon([[x, y], [x + w, y], [x + w, y + h], [x, y + h]]);
    setSelectionMode(false);
    setStatusMsg('Visible viewport selected.');
    setTimeout(handleAutoDetect, 100);
  };

  const handleAutoDetect = async () => {
    if (!selectedImage) return;
    setDetecting(true); setPoints([]); setStatusMsg('Scanning...'); setProgress(0);
    abortControllerRef.current = new AbortController();
    try {
      let url = `/api/detect-full?path=${encodeURIComponent(selectedImage.path)}&conf=0.1&tile_size=640&overlap=0.25`;
      if (polygon.length > 2) url += `&polygon=${encodeURIComponent(JSON.stringify(polygon))}`;
      const response = await fetch(url, { method: 'POST', signal: abortControllerRef.current.signal });
      const reader = response.body!.getReader(); const decoder = new TextDecoder(); let buffer = '';
      while (true) {
        const { value, done } = await reader.read(); if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n\n'); buffer = lines.pop() || '';
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          try {
            const data = JSON.parse(line.replace('data: ', ''));
            if (data.type === 'window') { setScanningBox(data.window); setProgress(data.progress || 0); }
            else if (data.type === 'points') setPoints(prev => [...prev, ...data.points]);
            else if (data.type === 'final') { setPoints(data.points); setScanningBox(null); setStatusMsg(`Found ${data.points.length} trees.`); }
          } catch (e) { }
        }
      }
    } catch (err: any) { setStatusMsg(err.name === 'AbortError' ? 'Stopped.' : 'Error'); }
    finally { setDetecting(false); setScanningBox(null); abortControllerRef.current = null; }
  };

  const handleSaveLabels = async () => {
    if (!selectedImage) return;
    try {
      await api.saveLabels(selectedImage.path, points);
      setStatusMsg('Labels saved successfully');
    } catch (err) { setStatusMsg('Failed to save labels'); }
  };

  const handleExportShapefile = async () => {
    if (!selectedImage || !points.length) return;
    setLoading('Exporting Shapefile...');
    try {
      const resp = await api.exportShapefile(selectedImage.path, points);
      setStatusMsg(`Saved: ${resp.data.path}`);
      console.log("Exported to:", resp.data.path);
    } catch (err) { setStatusMsg('Failed to export Shapefile'); console.error(err); }
    finally { setLoading(null); }
  };

  return (
    <div style={{ display: 'flex', height: '100vh', background: '#0a0a0a', color: '#e5e5e5', fontFamily: 'system-ui, sans-serif' }}>
      <div style={{ width: 300, background: '#111', borderRight: '1px solid #222', display: 'flex', flexDirection: 'column' }}>
        <div style={{ padding: 16, borderBottom: '1px solid #222', display: 'flex', alignItems: 'center', gap: 12 }}>
          <div style={{ width: 36, height: 36, background: '#16a34a', borderRadius: 8, display: 'flex', alignItems: 'center', justifyContent: 'center' }}><TreePine size={18} color="white" /></div>
          <div><div style={{ fontWeight: 700, fontSize: 16 }}>PalmAuto</div><div style={{ fontSize: 11, color: '#666' }}>Optimized Edition</div></div>
        </div>
        <div style={{ flex: 1, overflowY: 'auto', padding: '8px 0' }}>
          <div style={{ padding: '8px 16px', fontSize: 11, color: '#888', textTransform: 'uppercase', fontWeight: 600, display: 'flex', justifyContent: 'space-between' }}>
            <span>Images ({images.length})</span><button onClick={loadImages} style={{ background: 'transparent', border: 'none', color: '#666', cursor: 'pointer' }}>Refresh</button>
          </div>
          {images.map((img, i) => (
            <button key={i} onClick={() => handleSelectImage(img)} style={{
                display: 'block', width: '100%', textAlign: 'left', padding: '10px 16px',
                background: selectedImage?.path === img.path ? 'rgba(22,163,74,0.15)' : 'transparent',
                color: selectedImage?.path === img.path ? '#22c55e' : '#aaa',
                border: 'none', cursor: 'pointer', fontSize: 13, borderLeft: selectedImage?.path === img.path ? '3px solid #16a34a' : '3px solid transparent'
              }}><Layers size={14} style={{ display: 'inline', marginRight: 8, verticalAlign: 'middle' }} />{img.name}</button>
          ))}
        </div>
        {selectedImage && (
          <div style={{ padding: 16, borderTop: '1px solid #222', background: '#151515' }}>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, marginBottom: 12 }}>
              <div style={{ background: '#1a1a1a', padding: 12, borderRadius: 8, textAlign: 'center', border: '1px solid #222' }}><div style={{ fontSize: 22, fontWeight: 700, color: '#22c55e' }}>{points.length}</div><div style={{ fontSize: 10, color: '#666' }}>Trees</div></div>
              <div style={{ background: '#1a1a1a', padding: 12, borderRadius: 8, textAlign: 'center', border: '1px solid #222' }}><div style={{ fontSize: 11, color: '#888' }}>{selectedImage.width}x{selectedImage.height}</div><div style={{ fontSize: 10, color: '#666' }}>Resolution</div></div>
            </div>
            <div style={{ background: '#1a1a1a', borderRadius: 8, padding: 8, marginBottom: 12, border: '1px solid #222' }}>
              <div style={{ fontSize: 11, fontWeight: 600, color: '#888', marginBottom: 8, textAlign: 'center' }}>SELECTION (ROI)</div>
              {!selectionMode ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                  <button onClick={() => setSelectionMode(true)} style={{ width: '100%', padding: '8px', background: '#333', color: 'white', border: 'none', borderRadius: 6, cursor: 'pointer', fontSize: 11, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 4 }}>
                    <Crop size={14} /> Draw Polygon
                  </button>
                  <button onClick={handleDetectVisibleArea} style={{ width: '100%', padding: '8px', background: '#333', color: 'white', border: 'none', borderRadius: 6, cursor: 'pointer', fontSize: 11, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 4 }}>
                    <Eye size={14} /> Current Viewport
                  </button>
                </div>
              ) : (
                <div style={{ display: 'flex', gap: 4 }}>
                  <button onClick={() => setSelectionMode(false)} style={{ flex: 1, padding: '8px', background: '#16a34a', color: 'white', border: 'none', borderRadius: 6, cursor: 'pointer', fontSize: 11 }}>Finish</button>
                  <button onClick={() => {setPolygon([]); setRoiImage(null);}} style={{ padding: '8px', background: '#ef4444', color: 'white', border: 'none', borderRadius: 6, cursor: 'pointer' }}><RotateCcw size={14}/></button>
                </div>
              )}
              {polygon.length > 2 && !selectionMode && (
                <button onClick={handleCropToROI} style={{ width: '100%', marginTop: 8, padding: '10px', background: '#3b82f6', color: 'white', border: 'none', borderRadius: 6, cursor: 'pointer', fontSize: 11, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8 }}>
                  <ZoomIn size={14} /> Load High-Res
                </button>
              )}
            </div>
            <div style={{ background: '#1a1a1a', borderRadius: 8, padding: 8, marginBottom: 16, border: '1px solid #222' }}>
              <button onClick={handleAutoDetect} disabled={detecting} style={{ width: '100%', padding: '12px', background: '#16a34a', color: 'white', border: 'none', borderRadius: 6, fontWeight: 700, cursor: 'pointer', fontSize: 12, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8 }}>
                {!detecting ? <><Play size={14} fill="currentColor" /> Run Detection</> : <><X size={14} /> Stop ({Math.round(progress*100)}%)</>}
              </button>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              <button onClick={handleSaveLabels} style={{ padding: '8px', background: '#222', border: '1px solid #333', borderRadius: 6, color: '#ccc', cursor: 'pointer', fontSize: 11, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6 }}><Save size={14} /> Save Labels</button>
              <button onClick={handleExportShapefile} disabled={!points.length} style={{ padding: '8px', background: '#222', border: '1px solid #333', borderRadius: 6, color: '#ccc', cursor: 'pointer', fontSize: 11, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6 }}><Download size={14} /> Shapefile</button>
            </div>
          </div>
        )}
      </div>
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
        {selectedImage && (
          <div style={{ padding: '8px 16px', background: '#111', borderBottom: '1px solid #222', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}><div style={{ fontSize: 13, fontWeight: 500 }}>{selectedImage.name}</div></div>
            <div style={{ fontSize: 12 }}>
              {statusMsg && (
                <span style={{ 
                  color: statusMsg.toLowerCase().includes('failed') || statusMsg.toLowerCase().includes('error') ? '#ef4444' : '#22c55e', 
                  background: 'rgba(0,0,0,0.2)', 
                  padding: '4px 12px', 
                  borderRadius: 6,
                  display: 'flex',
                  alignItems: 'center',
                  gap: 6
                }}>
                  {statusMsg.toLowerCase().includes('success') ? <CheckCircle2 size={14}/> : <AlertCircle size={14}/>}
                  {statusMsg}
                </span>
              )}
            </div>
          </div>
        )}
        <div style={{ flex: 1, position: 'relative', overflow: 'hidden' }}>
          {selectedImage ? (
            <canvas ref={canvasRef} width={window.innerWidth - 300} height={window.innerHeight - 45}
              onWheel={handleWheel} onMouseDown={handleMouseDown} onMouseMove={handleMouseMove}
              onMouseUp={handleMouseUp} onMouseLeave={handleMouseUp} onClick={handleCanvasClick}
              onContextMenu={e => e.preventDefault()}
              style={{ display: 'block', cursor: selectionMode ? 'crosshair' : (isDragging.current ? 'grabbing' : 'grab') }} />
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', color: '#333' }}>
              <TreePine size={80} strokeWidth={1} /><div style={{ marginTop: 20, fontSize: 16 }}>Select an image to begin</div>
            </div>
          )}
        </div>
      </div>
      {loading && (
        <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.7)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1000, backdropFilter: 'blur(2px)' }}>
          <div style={{ background: '#1a1a1a', padding: '24px 40px', borderRadius: 12, border: '1px solid #333', textAlign: 'center' }}>
            <div style={{ width: 32, height: 32, border: '3px solid #333', borderTopColor: '#16a34a', borderRadius: '50%', animation: 'spin 1s linear infinite', margin: '0 auto 16px' }} />
            <div style={{ fontSize: 14 }}>{loading}</div>
          </div>
        </div>
      )}
      <style>{` @keyframes spin { to { transform: rotate(360deg); } } button:hover:not(:disabled) { filter: brightness(1.2); } canvas { touch-action: none; } `}</style>
    </div>
  );
};

export default App;
