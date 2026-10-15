// Browser ES module entry point. No bundler or build step required.
import { initNavigation } from './js/navigation.js?v=87';
import './js/uploads.js?v=87';
import { loadEnv } from './js/models.js?v=87';
import { loadImages, loadInferenceHistory } from './js/images.js?v=87';
import { loadVideos } from './js/video.js?v=87';
import { loadDatasets } from './js/training.js?v=87';
import { loadAnnotateGrid } from './js/annotator.js?v=87';
import { loadOutputs } from './js/outputs.js?v=87';
import { loadHeadCuts } from './js/head-cuts.js?v=87';
import { loadAggregate } from './js/aggregate.js?v=87';

initNavigation({ loadImages, loadInferenceHistory, loadVideos, loadDatasets, loadAnnotateGrid, loadHeadCuts, loadAggregate });
loadEnv();
loadImages();
loadOutputs();
