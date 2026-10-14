// Browser ES module entry point. No bundler or build step required.
import { initNavigation } from './js/navigation.js';
import './js/uploads.js';
import { loadEnv } from './js/models.js';
import { loadImages, loadInferenceHistory } from './js/images.js';
import { loadVideos } from './js/video.js';
import { loadDatasets } from './js/training.js';
import { loadAnnotateGrid } from './js/annotator.js';
import { loadOutputs } from './js/outputs.js';

initNavigation({ loadImages, loadInferenceHistory, loadVideos, loadDatasets, loadAnnotateGrid });
loadEnv();
loadImages();
loadOutputs();
