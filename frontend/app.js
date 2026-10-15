// Browser ES module entry point. No bundler or build step required.
import { initNavigation } from './js/navigation.js?v=85';
import './js/uploads.js?v=85';
import { loadEnv } from './js/models.js?v=85';
import { loadImages, loadInferenceHistory } from './js/images.js?v=85';
import { loadVideos } from './js/video.js?v=85';
import { loadDatasets } from './js/training.js?v=85';
import { loadAnnotateGrid } from './js/annotator.js?v=85';
import { loadOutputs } from './js/outputs.js?v=85';
import { loadHeadCuts } from './js/head-cuts.js?v=85';
import { loadAggregate } from './js/aggregate.js?v=85';

initNavigation({ loadImages, loadInferenceHistory, loadVideos, loadDatasets, loadAnnotateGrid, loadHeadCuts, loadAggregate });
loadEnv();
loadImages();
loadOutputs();
