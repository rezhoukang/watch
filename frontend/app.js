// Browser ES module entry point. No bundler or build step required.
import { initNavigation } from './js/navigation.js?v=71';
import './js/uploads.js?v=71';
import { loadEnv } from './js/models.js?v=71';
import { loadImages, loadInferenceHistory } from './js/images.js?v=71';
import { loadVideos } from './js/video.js?v=71';
import { loadDatasets } from './js/training.js?v=71';
import { loadAnnotateGrid } from './js/annotator.js?v=71';
import { loadOutputs } from './js/outputs.js?v=71';
import { loadHeadCuts } from './js/head-cuts.js?v=71';

initNavigation({ loadImages, loadInferenceHistory, loadVideos, loadDatasets, loadAnnotateGrid, loadHeadCuts });
loadEnv();
loadImages();
loadOutputs();
