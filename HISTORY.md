v0.2.2 / 2024-04-30
-------------------
* Add `PeakNearestTriggeringSalted` and `EventNearestTriggeringSalted` by @dachengx in https://github.com/dachengx/axidence/pull/47
* Set `child_plugin` to salting plugins by @dachengx in https://github.com/dachengx/axidence/pull/48
* Define `RunMeta` to help extract `start` and `end` of a run in the salting and pairing network by @dachengx in https://github.com/dachengx/axidence/pull/49
* Set `PeaksPaired` to be subclass of `DownChunkingPlugin` by @dachengx in https://github.com/dachengx/axidence/pull/50
* Apply Shadow Matching into `PairedPeaks` by @dachengx in https://github.com/dachengx/axidence/pull/51
* Add function to preprocess shadow related features by @dachengx in https://github.com/dachengx/axidence/pull/52
* Separate 2 and 3 hits pairing and add normalization factor by @dachengx in https://github.com/dachengx/axidence/pull/54
* Add `samplers` module, reweight S2 of salting result by @dachengx in https://github.com/dachengx/axidence/pull/56
* Implement only-S1/2 salting by @dachengx in https://github.com/dachengx/axidence/pull/57

**Full Changelog**: https://github.com/dachengx/axidence/compare/v0.2.1...v0.2.2


v0.2.1 / 2024-04-28
-------------------
* Use poetry by @dachengx in https://github.com/dachengx/axidence/pull/45

**Full Changelog**: https://github.com/dachengx/axidence/compare/v0.2.0...v0.2.1


v0.2.0 / 2024-04-27
-------------------
* Add `EventBuilding` cut by @dachengx in https://github.com/dachengx/axidence/pull/20
* Add prototype for isolated peaks by @dachengx in https://github.com/dachengx/axidence/pull/22
* Build events at once by @dachengx in https://github.com/dachengx/axidence/pull/27
* Save both peak and event-level fields into isolated peaks by @dachengx in https://github.com/dachengx/axidence/pull/30
* Prototype of `PairedPeaks` by @dachengx in https://github.com/dachengx/axidence/pull/31
* Add method `replication_tree` to `strax.Context` by @dachengx in https://github.com/dachengx/axidence/pull/34
* Connect salting and paring dependency tree by @dachengx in https://github.com/dachengx/axidence/pull/35
* Debug `do_compute` in plugin factory by @dachengx in https://github.com/dachengx/axidence/pull/37
* Accelerate `replication_tree` by @dachengx in https://github.com/dachengx/axidence/pull/38
* Add other peak and event level pairing plugins by @dachengx in https://github.com/dachengx/axidence/pull/39
* Add `PairingExists` by @dachengx in https://github.com/dachengx/axidence/pull/41

**Full Changelog**: https://github.com/dachengx/axidence/compare/v0.1.0...v0.2.0


v0.1.0 / 2024-04-24
-------------------
* Add package structure by @dachengx in https://github.com/dachengx/axidence/pull/6
* Add `ExhaustPlugin` to read chunks together at once by @dachengx in https://github.com/dachengx/axidence/pull/7
* Add Plugins to simulate Shadow, Ambience and SEDensity related features by @dachengx in https://github.com/dachengx/axidence/pull/8
* Add salted context and test dependency tree by @dachengx in https://github.com/dachengx/axidence/pull/9
* Add event building: `SaltedEvents` and `SaltedEventBasics` by @dachengx in https://github.com/dachengx/axidence/pull/10
* Assign features of salting events by @dachengx in https://github.com/dachengx/axidence/pull/11

**Full Changelog**: https://github.com/dachengx/axidence/compare/v0.0.0...v0.1.0


v0.0.0 / 2024-04-17
-------------------
* Add peripheral supporting files by @dachengx in https://github.com/dachengx/axidence/pull/1

New Contributors
* @dachengx made their first contribution in https://github.com/dachengx/axidence/pull/1

**Full Changelog**: https://github.com/dachengx/axidence/commits/v0.0.0
