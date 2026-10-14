// Browser ES module entry point. No bundler or build step required.
import { initNavigation } from './js/navigation.js?v=69';
import './js/uploads.js?v=69';
import { loadEnv } from './js/models.js?v=69';
import { loadImages, loadInferenceHistory } from './js/images.js?v=69';
import { loadVideos } from './js/video.js?v=69';
import { loadDatasets } from './js/training.js?v=69';
import { loadAnnotateGrid } from './js/annotator.js?v=69';
import { loadOutputs } from './js/outputs.js?v=69';

initNavigation({ loadImages, loadInferenceHistory, loadVideos, loadDatasets, loadAnnotateGrid });
loadEnv();
loadImages();
loadOutputs();
