v0.3.4 / 2025-05-18
-------------------
* Be compatible with strax >= 2 and straxen >= 3 by @dachengx in https://github.com/XENONnT/axidence/pull/96
* Validate the number of S1/2 group by @dachengx in https://github.com/XENONnT/axidence/pull/97
* Replace `hyperrun` with `superrun` by @dachengx in https://github.com/XENONnT/axidence/pull/99
* Compatible with https://github.com/XENONnT/straxen/pull/1532 by @dachengx in https://github.com/XENONnT/axidence/pull/100
* Remove CMT URLs by @dachengx in https://github.com/XENONnT/axidence/pull/101
* Compatible with https://github.com/XENONnT/straxen/pull/1513 by @dachengx in https://github.com/XENONnT/axidence/pull/98
* Inherit `center_time` from `peaklets` by @dachengx in https://github.com/XENONnT/axidence/pull/102
* `n_competing` plus one by @dachengx in https://github.com/XENONnT/axidence/pull/103
* Use numbered version of `docformatter` by @dachengx in https://github.com/XENONnT/axidence/pull/107

**Full Changelog**: https://github.com/XENONnT/axidence/compare/v0.3.3...v0.3.4


v0.3.3 / 2024-12-27
-------------------
* Poetry does not understand `requires-python` by @dachengx in https://github.com/XENONnT/axidence/pull/85
* Lock version of strax(en) in the test by @dachengx in https://github.com/XENONnT/axidence/pull/88
* Use SE scores instead of SE density by @dachengx in https://github.com/XENONnT/axidence/pull/87
* Fix bug in z simulation by @dachengx in https://github.com/XENONnT/axidence/pull/90
* Remove unused configs by @dachengx in https://github.com/XENONnT/axidence/pull/93
* Compatible with multi-dimensional field by @dachengx in https://github.com/XENONnT/axidence/pull/94

**Full Changelog**: https://github.com/XENONnT/axidence/compare/v0.3.2...v0.3.3


v0.3.2 / 2024-07-22
-------------------
* Allow saving of isolated cut by @dachengx in https://github.com/XENONnT/axidence/pull/82

**Full Changelog**: https://github.com/XENONnT/axidence/compare/v0.3.1...v0.3.2


v0.3.1 / 2024-05-27
-------------------
* Remove unused plugin by @dachengx in https://github.com/XENONnT/axidence/pull/76
* Add more tests with real data by @dachengx in https://github.com/XENONnT/axidence/pull/77
* Remove unnecessary duplicated tests by @dachengx in https://github.com/XENONnT/axidence/pull/78
* Set minimum drift length of salted events by @dachengx in https://github.com/XENONnT/axidence/pull/79
* Add structure of documentation by @dachengx in https://github.com/XENONnT/axidence/pull/80

**Full Changelog**: https://github.com/XENONnT/axidence/compare/v0.3.0...v0.3.1


v0.3.0 / 2024-05-14
-------------------
* Move `ExhaustPlugin` to strax by @dachengx in https://github.com/XENONnT/axidence/pull/59
* Separate `assign_plugin_attributes` for future usage by @dachengx in https://github.com/XENONnT/axidence/pull/60
* Add `get_paring_rate_correction` to `PeaksPaired` by @dachengx in https://github.com/XENONnT/axidence/pull/61
* Debug for attributes assignments by @dachengx in https://github.com/XENONnT/axidence/pull/62
* Add `run_meta` as the dependency of `PeaksPaired` and store `run_id` into `peaks_paired` by @dachengx in https://github.com/XENONnT/axidence/pull/63
* Sort `dtype_reference` to prevent dtype mismatch by @dachengx in https://github.com/XENONnT/axidence/pull/64
* Bug fix when isolated S2 in input buffer is 0 by @dachengx in https://github.com/XENONnT/axidence/pull/66
* Debug `data_kind` of `PairingExists` by @dachengx in https://github.com/XENONnT/axidence/pull/67
* Implement Hyperruns by @dachengx in https://github.com/XENONnT/axidence/pull/65
* Customized salting by @dachengx in https://github.com/XENONnT/axidence/pull/68
* Make chunk size samller by @dachengx in https://github.com/XENONnT/axidence/pull/69
* Simulate S2AFT by @dachengx in https://github.com/XENONnT/axidence/pull/70
* Combine salted events and salting events by @dachengx in https://github.com/XENONnT/axidence/pull/71
* Simplify `shadow_reference_selection` by @dachengx in https://github.com/XENONnT/axidence/pull/72
* Slove the bug when salting peaks overlap by @dachengx in https://github.com/XENONnT/axidence/pull/73
* Fix bug when triggering peak in an event is not from the salting peaks by @dachengx in https://github.com/XENONnT/axidence/pull/74

**Full Changelog**: https://github.com/XENONnT/axidence/compare/v0.2.2...v0.3.0


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
