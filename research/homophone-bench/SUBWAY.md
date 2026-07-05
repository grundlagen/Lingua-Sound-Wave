# Subway map: all hops indexed, hub nodes = interchange stations

Slime-mold/Tokyo-subway view of the chain-web: index every node across all 16,719
S-chains, count how many chains each sits in. High-count nodes = INTERCHANGES that
route everything. Cheap (one counting pass), flat output.

Files:
  subway-hubs.tsv    node \t n_chains (sorted) -- the stations
  subway-routes.txt  per top-400 hub: all its routes (the big flat collection)

Top interchanges:
  fr:dis 556 | fr:haines 498 | fr:sain 463 | fr:rient 422 | fr:met 411 ...

## Use: shortest route = via shared hub

Any two words that both appear in a hub's chain list are ~1 interchange apart.
The hub index IS the shortest-route matcher: to connect A and B, find a hub both
reach. No 7-hop search needed -- look up the table. arbre-type nodes that recur in
hundreds of chains are the transfer stations.

## Critical (Fable lens)

The hubs (dis=say, haines=hates, sain=healthy, met=puts) are common SHORT FR
fragments -- they interchange by SOUND, not sense. So this subway is the SOUND
backbone: maximal reach, meaning incidental. Great for the DATASET / connectivity
(every word routes), but routes through these hubs are sound-puns, not semantic.
For sensical translation pick MEANINGFUL hubs (content-word pivots with MUSE
meaning edges); for raw reach/dataset, the sound hubs are the efficient network.
Two maps: sound-subway (built, dense) and meaning-subway (needs the embedding/MUSE
meaning hubs) -- overlay them = homophonic+semantic routing.
