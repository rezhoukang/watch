// Browser ES module entry point. No bundler or build step required.
import { initNavigation } from './js/navigation.js?v=67';
import './js/uploads.js?v=67';
import { loadEnv } from './js/models.js?v=67';
import { loadImages, loadInferenceHistory } from './js/images.js?v=67';
import { loadVideos } from './js/video.js?v=67';
import { loadDatasets } from './js/training.js?v=67';
import { loadAnnotateGrid } from './js/annotator.js?v=67';
import { loadOutputs } from './js/outputs.js?v=67';

initNavigation({ loadImages, loadInferenceHistory, loadVideos, loadDatasets, loadAnnotateGrid });
loadEnv();
loadImages();
loadOutputs();
