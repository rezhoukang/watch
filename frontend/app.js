// Browser ES module entry point. No bundler or build step required.
import { initNavigation } from './js/navigation.js?v=90';
import './js/uploads.js?v=90';
import { loadEnv } from './js/models.js?v=90';
import { loadImages, loadInferenceHistory } from './js/images.js?v=90';
import { loadVideos } from './js/video.js?v=90';
import { loadDatasets } from './js/training.js?v=90';
import { loadAnnotateGrid } from './js/annotator.js?v=90';
import { loadOutputs } from './js/outputs.js?v=90';
import { loadHeadCuts } from './js/head-cuts.js?v=90';
import { loadAggregate } from './js/aggregate.js?v=90';

initNavigation({ loadImages, loadInferenceHistory, loadVideos, loadDatasets, loadAnnotateGrid, loadHeadCuts, loadAggregate });
loadEnv();
loadImages();
loadOutputs();
