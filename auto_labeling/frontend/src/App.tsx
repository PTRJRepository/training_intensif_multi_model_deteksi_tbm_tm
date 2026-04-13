import React, { useState, useEffect, useRef, useCallback } from 'react';
import { api, downloadShapefileFromBlob } from './api';
import { TreePine, Save, Download, Crop, Play, X, RotateCcw, ZoomIn, Eye, CheckCircle2, AlertCircle, Layers, EyeOff, Trash2, Upload, FolderOpen } from 'lucide-react';

export interface DetectionPoint { x: number; y: number; label: string; conf?: number; }
export interface ImageInfo { name: string; path: string; width: number; height: number; }
export interface Layer { id: string; name: string; points: DetectionPoint[]; visible: boolean; color: string; }

const App: React.FC = () => {
  const [images, setImages] = useState<any[]>([]);
  const [imageGroups, setImageGroups] = useState<{group: string, items: any[]}[]>([]);
  const [selectedImage, setSelectedImage] = useState<ImageInfo | null>(null);
  const [points, setPoints] = useState<DetectionPoint[]>([]);
  const [pointHistory, setPointHistory] = useState<DetectionPoint[][]>([]); // For undo
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

  // Manual point adding mode
  const [manualAddMode, setManualAddMode] = useState(false);
  const [selectedPointIdx, setSelectedPointIdx] = useState<number | null>(null);

  // Layer management
  const [layers, setLayers] = useState<Layer[]>([]);
  const [showLayerPanel, setShowLayerPanel] = useState(false);
  const [importLayerPath, setImportLayerPath] = useState('');
  const [showImportInput, setShowImportInput] = useState(false);

  // Detection settings
  const [detectSettings, setDetectSettings] = useState({
    conf: 0.1,
    tileSize: 640,
    overlap: 0.25,
    imgsz: 640
  });
  const [showSettings, setShowSettings] = useState(false);

  // Export settings
  const [exportDir, setExportDir] = useState('');
  const [showExportDirInput, setShowExportDirInput] = useState(false);
  const [tempExportDir, setTempExportDir] = useState('');
  const [viewportWidth, setViewportWidth] = useState(window.innerWidth);

  // Save history before changing points
  const saveToHistory = (currentPoints: DetectionPoint[]) => {
    setPointHistory(prev => [...prev.slice(-20), [...currentPoints]]); // Keep last 20 states
  };

  // Delete selected point
  const handleDeleteSelected = () => {
    if (selectedPointIdx === null) return;
    saveToHistory(points);
    setPoints(prev => prev.filter((_, i) => i !== selectedPointIdx));
    setSelectedPointIdx(null);
  };

  // Clear all manual points
  const handleClearManual = () => {
    saveToHistory(points);
    setPoints([]);
    setSelectedPointIdx(null);
  };

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [scale, setScale] = useState(1);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const scaleRef = useRef(1);
  const offsetRef = useRef({ x: 0, y: 0 });
  const [imgElement, setImgElement] = useState<HTMLImageElement | null>(null);
  const isDragging = useRef(false);
  const lastMousePos = useRef({ x: 0, y: 0 });

  useEffect(() => { loadImages(); }, []);
  useEffect(() => {
    const onResize = () => setViewportWidth(window.innerWidth);
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, []);

  const isCompact = viewportWidth <= 1100;

  useEffect(() => {
    scaleRef.current = scale;
  }, [scale]);

  useEffect(() => {
    offsetRef.current = offset;
  }, [offset]);
  const loadImages = async () => { 
    try { 
      const resp = await api.getImages(); 
      setImages(resp.data);
      
      // Group images by folder
      const groups: { [key: string]: any[] } = {};
      resp.data.forEach((img: any) => {
        const group = img.group || 'Other';
        if (!groups[group]) groups[group] = [];
        groups[group].push(img);
      });
      
      // Convert to sorted array
      const sortedGroups = Object.entries(groups)
        .map(([group, items]) => ({ group, items }))
        .sort((a, b) => a.group.localeCompare(b.group));
      setImageGroups(sortedGroups);
    } catch (err) { } 
    
    // Load export directory
    try {
      const dirResp = await api.getExportDir();
      setExportDir(dirResp.data.export_dir);
    } catch (err) { }
  };

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

    // Set canvas size first (before any transform)
    const displayWidth = canvas.clientWidth;
    const displayHeight = canvas.clientHeight;
    if (canvas.width !== displayWidth * dpr || canvas.height !== displayHeight * dpr) {
      canvas.width = displayWidth * dpr;
      canvas.height = displayHeight * dpr;
    }

    // Apply DPR scale - all drawing happens in CSS pixels
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.imageSmoothingEnabled = scale < 2.5;

    // Clear and fill background
    ctx.clearRect(0, 0, displayWidth, displayHeight);
    ctx.fillStyle = '#0a0a0a';
    ctx.fillRect(0, 0, displayWidth, displayHeight);

    // Apply world transform for image/detection drawing
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

    // Draw imported layers (behind main points)
    const layerRadius = Math.max(1.5, 4.0 / scale);
    layers.forEach(layer => {
      if (!layer.visible || layer.points.length === 0) return;
      
      // Parse layer color for fill and stroke
      ctx.fillStyle = layer.color;
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 1.0 / scale;
      
      layer.points.forEach((p: DetectionPoint) => {
        ctx.beginPath();
        ctx.arc(p.x, p.y, layerRadius, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      });
    });

    if (points.length > 0) {
      const radius = Math.max(1.2, 3.0 / scale);
      points.forEach((p, idx) => {
        const isSelected = idx === selectedPointIdx;
        const isManual = p.label === 'manual' || p.label === 'manual_tree';
        
        if (isManual) {
          // Manual points: yellow with blue outline
          ctx.fillStyle = '#eab308'; 
          ctx.strokeStyle = '#3b82f6';
          ctx.lineWidth = (isSelected ? 2.5 : 1.5) / scale;
        } else {
          // Detected points: green with white outline
          ctx.fillStyle = '#22c55e';
          ctx.strokeStyle = isSelected ? '#ef4444' : '#fff';
          ctx.lineWidth = (isSelected ? 2.5 : 0.8) / scale;
        }
        
        ctx.beginPath(); 
        ctx.arc(p.x, p.y, isSelected ? radius * 1.5 : radius, 0, Math.PI * 2); 
        ctx.fill(); 
        ctx.stroke();
        
        // Draw label badge for selected manual point
        if (isSelected && isManual) {
          ctx.fillStyle = '#3b82f6';
          ctx.beginPath();
          ctx.arc(p.x, p.y - radius - 3/scale, 5/scale, 0, Math.PI * 2);
          ctx.fill();
        }
      });
    }
    ctx.restore();
  }, [selectedImage, imgElement, points, layers, scale, offset, detecting, scanningBox, polygon, selectionMode, mousePos, roiImage, selectedPointIdx]);

  useEffect(() => { drawCanvas(); }, [drawCanvas]);

  const handleWheelNative = useCallback((e: WheelEvent) => {
    e.preventDefault();
    const canvas = canvasRef.current; if (!canvas) return;
    const factor = e.deltaY > 0 ? 0.85 : 1.15;
    const currentScale = scaleRef.current;
    const currentOffset = offsetRef.current;
    const newScale = Math.max(0.0001, Math.min(200, currentScale * factor));
    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left; const mouseY = e.clientY - rect.top;
    const worldX = (mouseX - currentOffset.x) / currentScale;
    const worldY = (mouseY - currentOffset.y) / currentScale;
    setOffset({ x: mouseX - worldX * newScale, y: mouseY - worldY * newScale }); setScale(newScale);
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.addEventListener('wheel', handleWheelNative, { passive: false });
    return () => canvas.removeEventListener('wheel', handleWheelNative);
  }, [handleWheelNative, selectedImage]);

  const handleMouseDown = (e: React.MouseEvent) => {
    const canvas = canvasRef.current; if (!canvas || !selectedImage) return;
    
    // Right-click = Pan mode
    if (e.button === 2) {
      isDragging.current = true;
      lastMousePos.current = { x: e.clientX, y: e.clientY };
      e.preventDefault();
      return;
    }
    
    if (selectionMode && e.button === 0) {
      const rect = canvas.getBoundingClientRect();
      const worldX = e.clientX - rect.left;
      const worldY = e.clientY - rect.top;
      const wx = (worldX - offset.x) / scale;
      const wy = (worldY - offset.y) / scale;
      setPolygon(prev => [...prev, [wx, wy]]); 
      return;
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
      const worldX = e.clientX - rect.left;
      const worldY = e.clientY - rect.top;
      const wx = (worldX - offset.x) / scale;
      const wy = (worldY - offset.y) / scale;
      setMousePos([wx, wy]);
    }
  };

  const handleMouseUp = () => { isDragging.current = false; };

  const handleCanvasClick = (e: React.MouseEvent) => {
    if (selectionMode || isDragging.current || e.button !== 0) return;
    const canvas = canvasRef.current; if (!canvas || !selectedImage) return;
    const rect = canvas.getBoundingClientRect();
    const worldX = e.clientX - rect.left;
    const worldY = e.clientY - rect.top;
    const x = (worldX - offset.x) / scale;
    const y = (worldY - offset.y) / scale;
    
    // Check if clicking near existing point
    const clickedIdx = points.findIndex(p => Math.hypot(p.x - x, p.y - y) < 15 / scale);
    
    if (manualAddMode) {
      // Manual add mode - add new manual point
      saveToHistory(points);
      setPoints(prev => [...prev, { x: Math.round(x), y: Math.round(y), label: 'manual', conf: 1.0 }]);
    } else if (clickedIdx !== -1) {
      // Click on existing point - select it
      setSelectedPointIdx(clickedIdx === selectedPointIdx ? null : clickedIdx);
    } else {
      // Click on empty space - deselect
      setSelectedPointIdx(null);
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
    // Save current points to history before detection
    saveToHistory(points);
    // Keep manual points separate
    const manualPoints = points.filter(p => p.label === 'manual' || p.label === 'manual_tree');
    
    setDetecting(true); setPoints(manualPoints); setStatusMsg('Scanning...'); setProgress(0);
    abortControllerRef.current = new AbortController();
    try {
      let url = `/api/detect-full?path=${encodeURIComponent(selectedImage.path)}&conf=${detectSettings.conf}&tile_size=${detectSettings.tileSize}&overlap=${detectSettings.overlap}&imgsz=${detectSettings.imgsz}`;
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
            else if (data.type === 'final') { 
              const detectedPoints = data.points.map((p: any) => ({...p, label: 'detected'}));
              setPoints([...manualPoints, ...detectedPoints]); 
              setScanningBox(null); 
              setStatusMsg(`Found ${detectedPoints.length} detected + ${manualPoints.length} manual`); 
            }
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
      const resp = await api.exportShapefile(selectedImage.path, points, exportDir);
      const savedPath = resp.data?.path || '';
      const savedFile = resp.data?.filename || '';
      const savedDir = resp.data?.directory || exportDir;
      const msg = savedPath
        ? `Saved shapefile: ${savedFile || savedPath}`
        : `Saved shapefile in ${savedDir}`;
      setStatusMsg(msg);
      console.log('Exported to:', savedPath || savedDir);
    } catch (err) { setStatusMsg('Failed to export Shapefile'); console.error(err); }
    finally { setLoading(null); }
  };

  const handleDownloadShapefile = async () => {
    if (!selectedImage || !points.length) return;
    setLoading('Preparing ZIP download...');
    try {
      const resp = await api.exportShapefileDownload(selectedImage.path, points, exportDir);
      const baseName = selectedImage.name.replace(/\.tiff?$/i, '');
      downloadShapefileFromBlob(resp.data, baseName);
      setStatusMsg('Shapefile ZIP downloaded');
    } catch (err) {
      setStatusMsg('Failed to download Shapefile ZIP');
      console.error(err);
    } finally {
      setLoading(null);
    }
  };

  const handlePickExportDir = async () => {
    setLoading('Opening folder picker...');
    try {
      const resp = await api.pickExportDir();
      const pickedDir = resp.data?.export_dir || '';
      if (resp.data?.status === 'cancelled') {
        setStatusMsg('Folder selection cancelled');
        return;
      }
      if (pickedDir) {
        setExportDir(pickedDir);
        setTempExportDir(pickedDir);
        setShowExportDirInput(true);
        setStatusMsg('Export dir selected');
      } else {
        setStatusMsg('No folder selected');
      }
    } catch (err) {
      setStatusMsg('Failed to open folder picker');
      console.error(err);
    } finally {
      setLoading(null);
    }
  };

  // Import shapefile as layer
  const handleImportLayer = async () => {
    if (!importLayerPath || !selectedImage) return;
    setLoading('Importing layer...');
    try {
      const resp = await api.importShapefile(importLayerPath, selectedImage.path);
      const layerName = importLayerPath.split(/[/\\]/).pop()?.replace('.shp', '') || 'layer';
      const newLayer: Layer = {
        id: Date.now().toString(),
        name: layerName,
        points: resp.data.points,
        visible: true,
        color: `hsl(${Math.random() * 360}, 70%, 60%)`
      };
      setLayers(prev => [...prev, newLayer]);
      setShowImportInput(false);
      setImportLayerPath('');
      setStatusMsg(`Imported ${resp.data.count} points as layer`);
    } catch (err) { setStatusMsg('Failed to import shapefile'); console.error(err); }
    finally { setLoading(null); }
  };

  // Toggle layer visibility
  const toggleLayerVisibility = (layerId: string) => {
    setLayers(prev => prev.map(l => l.id === layerId ? {...l, visible: !l.visible} : l));
  };

  // Remove layer
  const removeLayer = (layerId: string) => {
    setLayers(prev => prev.filter(l => l.id !== layerId));
  };

  return (
    <div style={{ display: 'flex', flexDirection: isCompact ? 'column' : 'row', height: '100vh', background: '#0a0a0a', color: '#e5e5e5', fontFamily: 'system-ui, sans-serif' }}>
      <div style={{ width: isCompact ? '100%' : 300, maxHeight: isCompact ? '42vh' : '100vh', background: '#111', borderRight: isCompact ? 'none' : '1px solid #222', borderTop: isCompact ? '1px solid #222' : 'none', display: 'flex', flexDirection: 'column', order: isCompact ? 2 : 1 }}>
        <div style={{ padding: 16, borderBottom: '1px solid #222', display: 'flex', alignItems: 'center', gap: 12 }}>
          <div style={{ width: 36, height: 36, background: '#16a34a', borderRadius: 8, display: 'flex', alignItems: 'center', justifyContent: 'center' }}><TreePine size={18} color="white" /></div>
          <div><div style={{ fontWeight: 700, fontSize: 16 }}>Rebinmas Tree Detection</div></div>
        </div>
        <div style={{ flex: 1, overflowY: 'auto', padding: '8px 0' }}>
          <div style={{ padding: '8px 16px', fontSize: 11, color: '#888', textTransform: 'uppercase', fontWeight: 600, display: 'flex', justifyContent: 'space-between' }}>
            <span>Images ({images.length})</span><button onClick={loadImages} style={{ background: 'transparent', border: 'none', color: '#666', cursor: 'pointer' }}>Refresh</button>
          </div>
          {imageGroups.map((grp, gi) => (
            <div key={gi} style={{ marginBottom: 4 }}>
              <div style={{ padding: '4px 16px', fontSize: 10, color: '#666', fontWeight: 600, background: '#1a1a1a' }}>
                {grp.group} ({grp.items.length})
              </div>
              {grp.items.map((img: any, i: number) => (
                <button key={i} onClick={() => handleSelectImage(img)} style={{
                    display: 'block', width: '100%', textAlign: 'left', padding: '8px 16px',
                    background: selectedImage?.path === img.path ? 'rgba(22,163,74,0.15)' : 'transparent',
                    color: selectedImage?.path === img.path ? '#22c55e' : '#aaa',
                    border: 'none', cursor: 'pointer', fontSize: 12, borderLeft: selectedImage?.path === img.path ? '3px solid #16a34a' : '3px solid transparent'
                  }}>
                  <span style={{ fontSize: 10, color: '#555', marginRight: 6 }}>{img.folder !== grp.group ? img.folder + '/' : ''}</span>
                  {img.name}
                </button>
              ))}
            </div>
          ))}
        </div>
        {selectedImage && (
          <div style={{ padding: 16, borderTop: '1px solid #222', background: '#151515', overflowY: 'auto' }}>
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
              <button onClick={() => setShowSettings(!showSettings)} style={{ width: '100%', marginTop: 6, padding: '6px', background: '#333', color: '#aaa', border: 'none', borderRadius: 4, cursor: 'pointer', fontSize: 10, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 4 }}>
                <Eye size={12} /> {showSettings ? 'Hide' : 'Show'} Settings
              </button>
              {showSettings && (
                <div style={{ marginTop: 8, display: 'flex', flexDirection: 'column', gap: 6, background: '#222', padding: 8, borderRadius: 4 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ fontSize: 10, color: '#888' }}>Confidence</span>
                    <span style={{ fontSize: 10, color: '#22c55e' }}>{detectSettings.conf}</span>
                  </div>
                  <input type="range" min="0.01" max="0.5" step="0.01" value={detectSettings.conf} onChange={e => setDetectSettings(s => ({...s, conf: parseFloat(e.target.value)}))} style={{ width: '100%' }} />
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ fontSize: 10, color: '#888' }}>Tile Size</span>
                    <span style={{ fontSize: 10, color: '#22c55e' }}>{detectSettings.tileSize}</span>
                  </div>
                  <input type="range" min="320" max="1280" step="32" value={detectSettings.tileSize} onChange={e => setDetectSettings(s => ({...s, tileSize: parseInt(e.target.value)}))} style={{ width: '100%' }} />
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ fontSize: 10, color: '#888' }}>Overlap</span>
                    <span style={{ fontSize: 10, color: '#22c55e' }}>{detectSettings.overlap}</span>
                  </div>
                  <input type="range" min="0" max="0.5" step="0.05" value={detectSettings.overlap} onChange={e => setDetectSettings(s => ({...s, overlap: parseFloat(e.target.value)}))} style={{ width: '100%' }} />
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ fontSize: 10, color: '#888' }}>Inference Size</span>
                    <span style={{ fontSize: 10, color: '#22c55e' }}>{detectSettings.imgsz}</span>
                  </div>
                  <input type="range" min="640" max="1920" step="64" value={detectSettings.imgsz} onChange={e => setDetectSettings(s => ({...s, imgsz: parseInt(e.target.value)}))} style={{ width: '100%' }} />
                </div>
              )}
            </div>

            {/* Layer Management */}
            <div style={{ background: '#1a1a1a', borderRadius: 8, padding: 8, marginBottom: 12, border: '1px solid #222' }}>
              <button 
                onClick={() => setShowLayerPanel(!showLayerPanel)} 
                style={{ width: '100%', padding: '8px', background: '#333', color: '#fff', border: 'none', borderRadius: 6, cursor: 'pointer', fontSize: 11, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 4 }}>
                <Layers size={14} /> Layers ({layers.length})
              </button>
              
              {showLayerPanel && (
                <div style={{ marginTop: 8 }}>
                  {/* Import Layer */}
                  <button 
                    onClick={() => setShowImportInput(!showImportInput)}
                    style={{ width: '100%', padding: '6px', background: '#16a34a', color: '#fff', border: 'none', borderRadius: 4, cursor: 'pointer', fontSize: 10, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 4, marginBottom: 6 }}>
                    <Upload size={12} /> Import Shapefile
                  </button>
                  
                  {showImportInput && (
                    <div style={{ background: '#222', padding: 8, borderRadius: 4, marginBottom: 6 }}>
                      <div style={{ fontSize: 9, color: '#666', marginBottom: 4 }}>Enter shapefile path (.shp):</div>
                      <input 
                        type="text" 
                        value={importLayerPath} 
                        onChange={e => setImportLayerPath(e.target.value)}
                        placeholder="D:\path\to\file.shp"
                        style={{ width: '100%', padding: '4px 6px', fontSize: 10, background: '#333', color: '#fff', border: '1px solid #444', borderRadius: 4 }}
                      />
                      <div style={{ display: 'flex', gap: 4, marginTop: 4 }}>
                        <button onClick={handleImportLayer} style={{ flex: 1, padding: '4px', background: '#16a34a', color: '#fff', border: 'none', borderRadius: 4, cursor: 'pointer', fontSize: 10 }}>Load</button>
                        <button onClick={() => setShowImportInput(false)} style={{ flex: 1, padding: '4px', background: '#666', color: '#fff', border: 'none', borderRadius: 4, cursor: 'pointer', fontSize: 10 }}>Cancel</button>
                      </div>
                    </div>
                  )}
                  
                  {/* Layer List */}
                  {layers.length === 0 ? (
                    <div style={{ fontSize: 10, color: '#666', textAlign: 'center', padding: 8 }}>No layers imported</div>
                  ) : (
                    <div style={{ maxHeight: 200, overflowY: 'auto' }}>
                      {layers.map(layer => (
                        <div key={layer.id} style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '6px 4px', background: '#222', borderRadius: 4, marginBottom: 4 }}>
                          <div 
                            onClick={() => toggleLayerVisibility(layer.id)}
                            style={{ 
                              width: 16, height: 16, borderRadius: 3, 
                              background: layer.color, 
                              cursor: 'pointer',
                              display: 'flex', alignItems: 'center', justifyContent: 'center',
                              opacity: layer.visible ? 1 : 0.4
                            }}>
                            {layer.visible ? <Eye size={10} color="#fff" /> : <EyeOff size={10} color="#fff" />}
                          </div>
                          <div style={{ flex: 1, fontSize: 10, color: '#ccc', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                            {layer.name} ({layer.points.length})
                          </div>
                          <button 
                            onClick={() => removeLayer(layer.id)}
                            style={{ padding: 2, background: 'transparent', border: 'none', cursor: 'pointer', color: '#ef4444' }}>
                            <Trash2 size={12} />
                          </button>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Editor Controls */}
            <div style={{ background: '#1a1a1a', borderRadius: 8, padding: 8, marginBottom: 12, border: '1px solid #222' }}>
              <div style={{ fontSize: 11, fontWeight: 600, color: '#888', marginBottom: 8, textAlign: 'center' }}>POINT EDITOR</div>
              
              {/* Manual Add Mode Toggle */}
              <button 
                onClick={() => { setManualAddMode(!manualAddMode); setSelectedPointIdx(null); }}
                style={{ 
                  width: '100%', padding: '8px', 
                  background: manualAddMode ? '#3b82f6' : '#333', 
                  color: 'white', border: 'none', borderRadius: 6, cursor: 'pointer', fontSize: 11, 
                  display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 4,
                  marginBottom: 6
                }}>
                <Crop size={14} /> {manualAddMode ? 'Exit Manual Mode' : 'Add Manual Point'}
              </button>
              
              {manualAddMode && (
                <div style={{ fontSize: 9, color: '#22c55e', textAlign: 'center', marginBottom: 4 }}>
                  Click on image to add point
                </div>
              )}

              {/* Stats */}
              <div style={{ display: 'flex', gap: 4, marginBottom: 6, fontSize: 9 }}>
                <div style={{ flex: 1, background: '#222', padding: '4px 6px', borderRadius: 4, textAlign: 'center' }}>
                  <span style={{ color: '#22c55e' }}>{points.filter(p => p.label !== 'manual' && p.label !== 'manual_tree').length}</span> detected
                </div>
                <div style={{ flex: 1, background: '#222', padding: '4px 6px', borderRadius: 4, textAlign: 'center' }}>
                  <span style={{ color: '#eab308' }}>{points.filter(p => p.label === 'manual' || p.label === 'manual_tree').length}</span> manual
                </div>
              </div>

              {/* Undo / Delete / Clear Controls */}
              <div style={{ display: 'flex', gap: 4 }}>
                <button 
                  onClick={() => { if (pointHistory.length > 0) { const prev = pointHistory[pointHistory.length - 1]; setPointHistory(h => h.slice(0, -1)); setPoints(prev); } }}
                  disabled={pointHistory.length === 0}
                  style={{ flex: 1, padding: '6px', background: '#333', color: pointHistory.length > 0 ? '#fff' : '#666', border: 'none', borderRadius: 4, cursor: pointHistory.length > 0 ? 'pointer' : 'not-allowed', fontSize: 10, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 4 }}>
                  <RotateCcw size={12} /> Undo
                </button>
                <button 
                  onClick={handleDeleteSelected}
                  disabled={selectedPointIdx === null}
                  style={{ flex: 1, padding: '6px', background: '#ef4444', color: selectedPointIdx !== null ? '#fff' : '#666', border: 'none', borderRadius: 4, cursor: selectedPointIdx !== null ? 'pointer' : 'not-allowed', fontSize: 10, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 4 }}>
                  <X size={12} /> Delete
                </button>
              </div>
              <button 
                onClick={() => { if (confirm('Clear all points?')) handleClearManual(); }}
                style={{ width: '100%', marginTop: 4, padding: '6px', background: '#333', color: '#ef4444', border: 'none', borderRadius: 4, cursor: 'pointer', fontSize: 10, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 4 }}>
                Clear All Points
              </button>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              <button onClick={handleSaveLabels} style={{ padding: '8px', background: '#222', border: '1px solid #333', borderRadius: 6, color: '#ccc', cursor: 'pointer', fontSize: 11, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6 }}><Save size={14} /> Save Labels</button>
              <button onClick={() => { setTempExportDir(exportDir); setShowExportDirInput(!showExportDirInput); }} style={{ padding: '8px', background: '#333', border: '1px solid #444', borderRadius: 6, color: '#ccc', cursor: 'pointer', fontSize: 10, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 4 }}>
                <Download size={12} /> Export Dir
              </button>
              <button onClick={handlePickExportDir} style={{ padding: '8px', background: '#1f2937', border: '1px solid #374151', borderRadius: 6, color: '#d1d5db', cursor: 'pointer', fontSize: 10, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 4 }}>
                <FolderOpen size={12} /> Browse Folder
              </button>
              {showExportDirInput && (
                <div style={{ background: '#222', padding: 8, borderRadius: 4 }}>
                  <div style={{ fontSize: 9, color: '#666', marginBottom: 4 }}>Current: {exportDir}</div>
                  <input 
                    type="text" 
                    value={tempExportDir} 
                    onChange={e => setTempExportDir(e.target.value)}
                    placeholder="Enter export path..."
                    style={{ width: '100%', padding: '4px 6px', fontSize: 10, background: '#333', color: '#fff', border: '1px solid #444', borderRadius: 4 }}
                  />
                  <div style={{ display: 'flex', gap: 4, marginTop: 4 }}>
                    <button onClick={handlePickExportDir} style={{ flex: 1, padding: '4px', background: '#1f2937', color: '#d1d5db', border: '1px solid #374151', borderRadius: 4, cursor: 'pointer', fontSize: 10 }}>Browse</button>
                    <button onClick={async () => { try { await api.setExportDir(tempExportDir); setExportDir(tempExportDir); setShowExportDirInput(false); setStatusMsg('Export dir updated'); } catch { setStatusMsg('Failed to set dir'); } }} style={{ flex: 1, padding: '4px', background: '#16a34a', color: 'white', border: 'none', borderRadius: 4, cursor: 'pointer', fontSize: 10 }}>Set</button>
                    <button onClick={() => setShowExportDirInput(false)} style={{ flex: 1, padding: '4px', background: '#ef4444', color: 'white', border: 'none', borderRadius: 4, cursor: 'pointer', fontSize: 10 }}>Cancel</button>
                  </div>
                </div>
              )}
              <button onClick={handleExportShapefile} disabled={!points.length} style={{ padding: '8px', background: '#222', border: '1px solid #333', borderRadius: 6, color: '#ccc', cursor: 'pointer', fontSize: 11, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6 }}><Download size={14} /> Save Shapefile</button>
              <button onClick={handleDownloadShapefile} disabled={!points.length} style={{ padding: '8px', background: '#1f2937', border: '1px solid #374151', borderRadius: 6, color: '#d1d5db', cursor: 'pointer', fontSize: 11, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6 }}><Download size={14} /> Download ZIP</button>
            </div>
          </div>
        )}
      </div>
      <div style={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column', order: isCompact ? 1 : 2 }}>
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
            <canvas ref={canvasRef}
              onMouseDown={handleMouseDown} onMouseMove={handleMouseMove}
              onMouseUp={handleMouseUp} onMouseLeave={handleMouseUp} onClick={handleCanvasClick}
              onContextMenu={e => e.preventDefault()}
              style={{ display: 'block', cursor: (selectionMode || manualAddMode) ? 'crosshair' : (isDragging.current ? 'grabbing' : 'grab'), width: '100%', height: '100%' }} />
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
