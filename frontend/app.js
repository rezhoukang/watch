// Browser ES module entry point. No bundler or build step required.
import { initNavigation } from './js/navigation.js?v=82';
import './js/uploads.js?v=82';
import { loadEnv } from './js/models.js?v=82';
import { loadImages, loadInferenceHistory } from './js/images.js?v=82';
import { loadVideos } from './js/video.js?v=82';
import { loadDatasets } from './js/training.js?v=82';
import { loadAnnotateGrid } from './js/annotator.js?v=82';
import { loadOutputs } from './js/outputs.js?v=82';
import { loadHeadCuts } from './js/head-cuts.js?v=82';

initNavigation({ loadImages, loadInferenceHistory, loadVideos, loadDatasets, loadAnnotateGrid, loadHeadCuts });
loadEnv();
loadImages();
loadOutputs();
