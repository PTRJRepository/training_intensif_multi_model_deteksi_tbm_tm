import axios from 'axios';

const API_BASE = '/api';

export interface ModelInfo {
  id: string;
  name: string;
  label: string;
  class_tag: 'tbm' | 'tm';
  model_tag: string;
  path: string;
  resolved_path: string;
  enabled: boolean;
  conf: number;
  imgsz: number;
  exists: boolean;
  loaded: boolean;
  modified_at?: string | null;
  modified_ts?: number | null;
  error?: string | null;
}

export interface ModelsResponse {
  models: ModelInfo[];
  loaded_model_ids: string[];
  errors: Record<string, string>;
  config_path: string;
  auto_best_by_class: Record<string, string | null>;
}

const serializeModelIds = (modelIds?: string[]) => {
  if (!modelIds || modelIds.length === 0) {
    return '';
  }
  return `&model_ids=${encodeURIComponent(modelIds.join(','))}`;
};

type FullDetectionUrlParams = {
  path: string;
  conf?: number;
  tileSize?: number;
  overlap?: number;
  imgsz?: number;
  modelIds?: string[];
  zoomScales?: string;
  dbscanEps?: number;
  clusterMethod?: string;
  maxPasses?: number;
  minNewPoints?: number;
  batchSize?: number;
  inferClass?: 'tbm' | 'tm';
  polygon?: [number, number][];
};

export const buildDetectFullUrl = ({
  path,
  conf = 0.1,
  tileSize = 640,
  overlap = 0.25,
  imgsz = 640,
  modelIds,
  zoomScales,
  dbscanEps = 12.0,
  clusterMethod = 'hybrid',
  maxPasses = 2,
  minNewPoints = 2,
  batchSize = 8,
  inferClass = 'tbm',
  polygon,
}: FullDetectionUrlParams) => {
  const params = new URLSearchParams({
    path,
    conf: String(conf),
    tile_size: String(tileSize),
    overlap: String(overlap),
    imgsz: String(imgsz),
    infer_class: inferClass,
    dbscan_eps: String(dbscanEps),
    cluster_method: clusterMethod,
    max_passes: String(maxPasses),
    min_new_points: String(minNewPoints),
    batch_size: String(batchSize),
  });

  if (modelIds && modelIds.length > 0) {
    params.set('model_ids', modelIds.join(','));
  }

  if (zoomScales && zoomScales.trim()) {
    params.set('zoom_scales', zoomScales.trim());
  }

  if (polygon && polygon.length > 2) {
    params.set('polygon', JSON.stringify(polygon));
  }

  return `${API_BASE}/detect-full?${params.toString()}`;
};

export const api = {
  getImages: () => axios.get(`${API_BASE}/images`),
  getImageInfo: (path: string) => axios.get(`${API_BASE}/image-info?path=${encodeURIComponent(path)}`),
  getModels: () => axios.get<ModelsResponse>(`${API_BASE}/models`),
  reloadModels: () => axios.post<ModelsResponse & { status: string }>(`${API_BASE}/models/reload`),
  getTileUrl: (path: string, x: number, y: number, w: number, h: number, scale: number = 1.0) =>
    `${API_BASE}/tile?path=${encodeURIComponent(path)}&x=${x}&y=${y}&w=${w}&h=${h}&scale=${scale}`,
  runDetection: (
    path: string,
    x: number,
    y: number,
    w: number,
    h: number,
    conf?: number,
    imgsz?: number,
    modelIds?: string[],
    inferClass: 'tbm' | 'tm' = 'tbm',
  ) =>
    axios.post(
      `${API_BASE}/detect?path=${encodeURIComponent(path)}&x=${x}&y=${y}&w=${w}&h=${h}&conf=${conf ?? 0.1}&imgsz=${imgsz ?? 640}&infer_class=${inferClass}${serializeModelIds(modelIds)}`
    ),
  runFullDetection: (
    path: string,
    conf?: number,
    tileSize?: number,
    overlap?: number,
    imgsz?: number,
    modelIds?: string[],
    zoomScales?: string,
    dbscanEps?: number,
    clusterMethod: string = 'hybrid',
    maxPasses: number = 2,
    minNewPoints: number = 2,
    batchSize: number = 8,
    inferClass: 'tbm' | 'tm' = 'tbm',
  ) =>
    axios.post(
      buildDetectFullUrl({
        path,
        conf,
        tileSize,
        overlap,
        imgsz,
        modelIds,
        zoomScales,
        dbscanEps,
        clusterMethod,
        maxPasses,
        minNewPoints,
        batchSize,
        inferClass,
      })
    ),
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
