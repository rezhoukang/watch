// Browser ES module entry point. No bundler or build step required.
import { initNavigation } from './js/navigation.js?v=95';
import './js/uploads.js?v=95';
import { loadEnv } from './js/models.js?v=95';
import { loadImages, loadInferenceHistory } from './js/images.js?v=95';
import { loadVideos } from './js/video.js?v=95';
import { loadDatasets } from './js/training.js?v=95';
import { loadAnnotateGrid } from './js/annotator.js?v=95';
import { loadOutputs } from './js/outputs.js?v=95';
import { loadHeadCuts } from './js/head-cuts.js?v=95';
import { loadAggregate } from './js/aggregate.js?v=101';
import { loadActions } from './js/actions.js?v=98';

initNavigation({ loadImages, loadInferenceHistory, loadVideos, loadDatasets, loadAnnotateGrid, loadHeadCuts, loadAggregate, loadActions });
loadEnv();
loadImages();
loadOutputs();
