// Browser ES module entry point. No bundler or build step required.
import { initNavigation } from './js/navigation.js?v=83';
import './js/uploads.js?v=83';
import { loadEnv } from './js/models.js?v=83';
import { loadImages, loadInferenceHistory } from './js/images.js?v=83';
import { loadVideos } from './js/video.js?v=83';
import { loadDatasets } from './js/training.js?v=83';
import { loadAnnotateGrid } from './js/annotator.js?v=83';
import { loadOutputs } from './js/outputs.js?v=83';
import { loadHeadCuts } from './js/head-cuts.js?v=83';
import { loadAggregate } from './js/aggregate.js?v=83';

initNavigation({ loadImages, loadInferenceHistory, loadVideos, loadDatasets, loadAnnotateGrid, loadHeadCuts, loadAggregate });
loadEnv();
loadImages();
loadOutputs();
