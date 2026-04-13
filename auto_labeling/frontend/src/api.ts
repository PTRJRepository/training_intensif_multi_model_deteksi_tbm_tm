import axios from 'axios';

const API_BASE = '/api';

export const api = {
  getImages: () => axios.get(`${API_BASE}/images`),
  getImageInfo: (path: string) => axios.get(`${API_BASE}/image-info?path=${encodeURIComponent(path)}`),
  getTileUrl: (path: string, x: number, y: number, w: number, h: number, scale: number = 1.0) =>
    `${API_BASE}/tile?path=${encodeURIComponent(path)}&x=${x}&y=${y}&w=${w}&h=${h}&scale=${scale}`,
  runDetection: (path: string, x: number, y: number, w: number, h: number, conf?: number, imgsz?: number) =>
    axios.post(`${API_BASE}/detect?path=${encodeURIComponent(path)}&x=${x}&y=${y}&w=${w}&h=${h}&conf=${conf || 0.1}&imgsz=${imgsz || 640}`),
  runFullDetection: (path: string, conf?: number, tileSize?: number, overlap?: number, imgsz?: number) =>
    axios.post(`${API_BASE}/detect-full?path=${encodeURIComponent(path)}&conf=${conf || 0.1}&tile_size=${tileSize || 640}&overlap=${overlap || 0.25}&imgsz=${imgsz || 640}`),
  saveLabels: (path: string, labels: any[]) =>
    axios.post(`${API_BASE}/save?path=${encodeURIComponent(path)}`, labels),
  loadLabels: (path: string) =>
    axios.get(`${API_BASE}/load?path=${encodeURIComponent(path)}`),
  getExportDir: () => axios.get(`${API_BASE}/export-dir`),
  pickExportDir: () => axios.get(`${API_BASE}/export-dir/pick`),
  setExportDir: (exportDir: string) => axios.post(`${API_BASE}/export-dir`, exportDir),
  exportDataset: (path: string, labels: any[], exportDir?: string) =>
    axios.post(`${API_BASE}/export/dataset?path=${encodeURIComponent(path)}&output_dir=${encodeURIComponent(exportDir || '')}`, labels),
  importShapefile: (shpPath: string, refImagePath: string) =>
    axios.get(`${API_BASE}/import/shapefile?shp_path=${encodeURIComponent(shpPath)}&ref_image=${encodeURIComponent(refImagePath)}`),
  exportShapefile: (path: string, labels: any[], exportDir?: string) =>
    axios.post(
      `${API_BASE}/export/shapefile?path=${encodeURIComponent(path)}&output_dir=${encodeURIComponent(exportDir || '')}`,
      labels
    ),
  exportShapefileDownload: (path: string, labels: any[], exportDir?: string) =>
    axios.post(
      `${API_BASE}/export/shapefile/download?path=${encodeURIComponent(path)}&output_dir=${encodeURIComponent(exportDir || '')}`,
      labels,
      { responseType: 'blob' }
    ),
};

// Helper to download shapefile as ZIP
export const downloadShapefile = async (path: string, labels: any[]) => {
  const response = await axios.post(
    `${API_BASE}/export/shapefile?path=${encodeURIComponent(path)}`,
    labels,
    { responseType: 'blob' }
  );
  
  // Create download link
  const url = window.URL.createObjectURL(new Blob([response.data]));
  const link = document.createElement('a');
  link.href = url;
  const baseName = path.split(/[/\\]/).pop()?.replace('.tif', '').replace('.tiff', '') || 'export';
  link.download = `${baseName}_labels.zip`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  window.URL.revokeObjectURL(url);
};

export const downloadShapefileFromBlob = (blob: Blob, baseName: string) => {
  const url = window.URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `${baseName}_labels.zip`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  window.URL.revokeObjectURL(url);
};
