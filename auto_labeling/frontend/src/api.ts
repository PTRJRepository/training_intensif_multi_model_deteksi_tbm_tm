import axios from 'axios';

const API_BASE = '/api';

export const api = {
  getImages: () => axios.get(`${API_BASE}/images`),
  getImageInfo: (path: string) => axios.get(`${API_BASE}/image-info?path=${encodeURIComponent(path)}`),
  getTileUrl: (path: string, x: number, y: number, w: number, h: number, scale: number = 1.0) =>
    `${API_BASE}/tile?path=${encodeURIComponent(path)}&x=${x}&y=${y}&w=${w}&h=${h}&scale=${scale}`,
  runDetection: (path: string, x: number, y: number, w: number, h: number) =>
    axios.post(`${API_BASE}/detect?path=${encodeURIComponent(path)}&x=${x}&y=${y}&w=${w}&h=${h}`),
  runFullDetection: (path: string, conf?: number, tileSize?: number, overlap?: number) =>
    axios.post(`${API_BASE}/detect-full?path=${encodeURIComponent(path)}&conf=${conf || 0.01}&tile_size=${tileSize || 1280}&overlap=${overlap || 200}`),
  saveLabels: (path: string, labels: any[]) =>
    axios.post(`${API_BASE}/save?path=${encodeURIComponent(path)}`, labels),
  loadLabels: (path: string) =>
    axios.get(`${API_BASE}/load?path=${encodeURIComponent(path)}`),
  exportShapefile: (path: string, labels: any[]) =>
    axios.post(`${API_BASE}/export/shapefile?path=${encodeURIComponent(path)}`, labels),
  exportDataset: (path: string, labels: any[]) =>
    axios.post(`${API_BASE}/export/dataset?path=${encodeURIComponent(path)}`, labels),
};
