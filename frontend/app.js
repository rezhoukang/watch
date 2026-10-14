// Browser ES module entry point. No bundler or build step required.
import { initNavigation } from './js/navigation.js?v=66';
import './js/uploads.js?v=66';
import { loadEnv } from './js/models.js?v=66';
import { loadImages, loadInferenceHistory } from './js/images.js?v=66';
import { loadVideos } from './js/video.js?v=66';
import { loadDatasets } from './js/training.js?v=66';
import { loadAnnotateGrid } from './js/annotator.js?v=66';
import { loadOutputs } from './js/outputs.js?v=66';

initNavigation({ loadImages, loadInferenceHistory, loadVideos, loadDatasets, loadAnnotateGrid });
loadEnv();
loadImages();
loadOutputs();
