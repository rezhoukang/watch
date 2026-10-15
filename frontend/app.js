// Browser ES module entry point. No bundler or build step required.
import { initNavigation } from './js/navigation.js?v=86';
import './js/uploads.js?v=86';
import { loadEnv } from './js/models.js?v=86';
import { loadImages, loadInferenceHistory } from './js/images.js?v=86';
import { loadVideos } from './js/video.js?v=86';
import { loadDatasets } from './js/training.js?v=86';
import { loadAnnotateGrid } from './js/annotator.js?v=86';
import { loadOutputs } from './js/outputs.js?v=86';
import { loadHeadCuts } from './js/head-cuts.js?v=86';
import { loadAggregate } from './js/aggregate.js?v=86';

initNavigation({ loadImages, loadInferenceHistory, loadVideos, loadDatasets, loadAnnotateGrid, loadHeadCuts, loadAggregate });
loadEnv();
loadImages();
loadOutputs();
