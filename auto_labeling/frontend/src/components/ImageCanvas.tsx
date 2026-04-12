import React, { useRef, useEffect, useState, useCallback } from 'react';
import { api } from '../api';
import type { ImageInfo, DetectionPoint } from '../App';
import { ZoomIn, ZoomOut, RefreshCw, Crosshair } from 'lucide-react';

interface Props {
  imageInfo: ImageInfo;
  points: DetectionPoint[];
  setPoints: React.Dispatch<React.SetStateAction<DetectionPoint[]>>;
  onAutoDetect: (view: { x: number; y: number; w: number; h: number }) => void;
  sidebarOpen?: boolean;
}

const ImageCanvas: React.FC<Props> = ({ imageInfo, points, setPoints, onAutoDetect, sidebarOpen = true }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [scale, setScale] = useState(1);
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const [imgElement, setImgElement] = useState<HTMLImageElement | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const sidebarWidth = sidebarOpen ? 280 : 0;

  // Load image when imageInfo changes
  useEffect(() => {
    setImgElement(null);
    setScale(1);

    const tileUrl = api.getTileUrl(imageInfo.path, 0, 0, imageInfo.width, imageInfo.height, 0.5);
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.src = tileUrl;
    img.onload = () => {
      setImgElement(img);
      // Center image
      const canvas = canvasRef.current;
      if (canvas) {
        const s = Math.min((canvas.width - 40) / imageInfo.width, (canvas.height - 40) / imageInfo.height) * 0.9;
        setScale(s);
        setPosition({
          x: (canvas.width - imageInfo.width * s) / 2,
          y: (canvas.height - imageInfo.height * s) / 2
        });
      }
    };
  }, [imageInfo.path]);

  // Draw canvas
  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;

    const toolbarH = 50;
    canvas.width = window.innerWidth - sidebarWidth;
    canvas.height = window.innerHeight - toolbarH;

    ctx.fillStyle = '#0a0a0a';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    if (imgElement) {
      ctx.save();
      ctx.translate(position.x, position.y);
      ctx.scale(scale, scale);
      ctx.drawImage(imgElement, 0, 0);
      ctx.restore();
    }

    // Draw points
    if (points.length > 0) {
      const r = Math.max(2, 5 / scale);
      ctx.save();
      ctx.translate(position.x, position.y);
      ctx.scale(scale, scale);

      ctx.fillStyle = '#22c55e';
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 1.5 / scale;

      ctx.beginPath();
      points.forEach(p => {
        ctx.moveTo(p.x + r, p.y);
        ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
      });
      ctx.fill();
      ctx.stroke();
      ctx.restore();
    }
  }, [imgElement, points, scale, position, sidebarWidth]);

  useEffect(() => {
    draw();
    window.addEventListener('resize', draw);
    return () => window.removeEventListener('resize', draw);
  }, [draw]);

  // Mouse wheel zoom
  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const factor = e.deltaY > 0 ? 0.9 : 1.1;
    setScale(s => Math.max(0.05, Math.min(30, s * factor)));
  };

  // Click to add/remove point
  const handleClick = (e: React.MouseEvent) => {
    if (isDragging) return;

    const canvas = canvasRef.current;
    if (!canvas || !imageInfo) return;

    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left - position.x) / scale;
    const y = (e.clientY - rect.top - position.y) / scale;

    if (e.button === 2) {
      // Right click - remove nearest
      const idx = points.findIndex((p: DetectionPoint) => Math.hypot(p.x - x, p.y - y) < 15 / scale);
      if (idx !== -1) setPoints((prev: DetectionPoint[]) => prev.filter((_: DetectionPoint, i: number) => i !== idx));
    } else {
      // Left click - add point
      setPoints((prev: DetectionPoint[]) => [...prev, { x: Math.round(x), y: Math.round(y), label: 'palm_tree', conf: 1 }]);
    }
  };

  // Pan
  const handleMouseDown = (e: React.MouseEvent) => {
    if (e.button === 0) setIsDragging(false);
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (e.buttons === 1 && !isDragging) {
      setPosition(p => ({ x: p.x + e.movementX, y: p.y + e.movementY }));
    }
  };

  const handleMouseUp = () => setIsDragging(false);

  const loadHighRes = () => {
    const tileUrl = api.getTileUrl(imageInfo.path, 0, 0, imageInfo.width, imageInfo.height, 1);
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.src = tileUrl;
    img.onload = () => setImgElement(img);
  };

  const triggerAutoDetect = () => {
    const canvas = canvasRef.current;
    if (!canvas || !imageInfo) return;
    const s = scale;
    const x = Math.max(0, Math.floor(-position.x / s));
    const y = Math.max(0, Math.floor(-position.y / s));
    const w = Math.min(imageInfo.width, Math.ceil(canvas.width / s));
    const h = Math.min(imageInfo.height, Math.ceil(canvas.height / s));
    onAutoDetect({ x, y, w, h });
  };

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%', overflow: 'hidden', background: '#0a0a0a' }}>
      <canvas
        ref={canvasRef}
        onWheel={handleWheel}
        onClick={handleClick}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onContextMenu={e => e.preventDefault()}
        style={{ display: 'block', cursor: 'crosshair' }}
      />

      {/* Controls - Bottom Right */}
      <div style={{ position: 'absolute', bottom: 16, right: 16, display: 'flex', flexDirection: 'column', gap: 8 }}>
        <button
          onClick={triggerAutoDetect}
          style={{
            display: 'flex', alignItems: 'center', gap: 8, padding: '10px 16px',
            background: '#16a34a', color: 'white', border: 'none', borderRadius: 8,
            fontWeight: 600, cursor: 'pointer', fontSize: 13, boxShadow: '0 4px 12px rgba(0,0,0,0.4)'
          }}
        >
          <Crosshair size={16} /> Detect View
        </button>

        <div style={{ background: '#1a1a1a', borderRadius: 8, border: '1px solid #333', overflow: 'hidden' }}>
          <button onClick={() => setScale(s => s * 1.3)} style={{ width: 40, height: 40, background: 'transparent', border: 'none', cursor: 'pointer', color: '#ccc' }}>
            <ZoomIn size={18} />
          </button>
          <button onClick={() => setScale(s => s / 1.3)} style={{ width: 40, height: 40, background: 'transparent', border: 'none', borderTop: '1px solid #333', cursor: 'pointer', color: '#ccc' }}>
            <ZoomOut size={18} />
          </button>
          <button onClick={loadHighRes} style={{ width: 40, height: 40, background: 'transparent', border: 'none', borderTop: '1px solid #333', cursor: 'pointer', color: '#22c55e' }}>
            <RefreshCw size={18} />
          </button>
        </div>
      </div>

      {/* Info - Bottom Left */}
      <div style={{ position: 'absolute', bottom: 16, left: 16, display: 'flex', gap: 8 }}>
        <div style={{ background: '#1a1a1a', padding: '6px 12px', borderRadius: 6, border: '1px solid #333', fontSize: 12 }}>
          <span style={{ color: '#22c55e', fontWeight: 600 }}>{Math.round(scale * 100)}%</span>
        </div>
        <div style={{ background: '#1a1a1a', padding: '6px 12px', borderRadius: 6, border: '1px solid #333', fontSize: 12, color: '#888' }}>
          <span style={{ color: '#fff' }}>{points.length}</span> trees
        </div>
      </div>

      {/* Instructions - Top Left */}
      <div style={{ position: 'absolute', top: 12, left: 12, background: '#1a1a1a', padding: '6px 10px', borderRadius: 6, fontSize: 11, color: '#666', border: '1px solid #222' }}>
        Left: Add • Right: Remove • Drag: Pan
      </div>
    </div>
  );
};

export default ImageCanvas;