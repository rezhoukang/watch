// Browser ES module entry point. No bundler or build step required.
import { initNavigation } from './js/navigation.js?v=94';
import './js/uploads.js?v=94';
import { loadEnv } from './js/models.js?v=94';
import { loadImages, loadInferenceHistory } from './js/images.js?v=94';
import { loadVideos } from './js/video.js?v=94';
import { loadDatasets } from './js/training.js?v=94';
import { loadAnnotateGrid } from './js/annotator.js?v=94';
import { loadOutputs } from './js/outputs.js?v=94';
import { loadHeadCuts } from './js/head-cuts.js?v=94';
import { loadAggregate } from './js/aggregate.js?v=94';
import { loadActions } from './js/actions.js?v=94';

initNavigation({ loadImages, loadInferenceHistory, loadVideos, loadDatasets, loadAnnotateGrid, loadHeadCuts, loadAggregate, loadActions });
loadEnv();
loadImages();
loadOutputs();
