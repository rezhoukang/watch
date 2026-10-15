// Browser ES module entry point. No bundler or build step required.
import { initNavigation } from './js/navigation.js?v=88';
import './js/uploads.js?v=88';
import { loadEnv } from './js/models.js?v=88';
import { loadImages, loadInferenceHistory } from './js/images.js?v=88';
import { loadVideos } from './js/video.js?v=88';
import { loadDatasets } from './js/training.js?v=88';
import { loadAnnotateGrid } from './js/annotator.js?v=88';
import { loadOutputs } from './js/outputs.js?v=88';
import { loadHeadCuts } from './js/head-cuts.js?v=88';
import { loadAggregate } from './js/aggregate.js?v=88';

initNavigation({ loadImages, loadInferenceHistory, loadVideos, loadDatasets, loadAnnotateGrid, loadHeadCuts, loadAggregate });
loadEnv();
loadImages();
loadOutputs();
