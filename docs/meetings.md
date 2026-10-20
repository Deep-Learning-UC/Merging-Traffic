meeting 4 10/20/26
 - gravitate towards stable baselines
 - expand MVP - what we have is too simple
   - due 11/15 reminder - 3 weeks and 5 days
   - expand personalities, and adjust based on personalities
 - jace has a script - different configurations for scenarios
   - train on simple
   - train on expanded case - e.g. rush hour
   - alter AI for vehicles / personality
 - is it possible to adjust the model for e.g. changing to left lane instead of right lane? - for adding complexity
 - model should be able to get to the end as quickly as possible - time element
 - compare different ones initially, the pick one eventually
 - first narrow down to which models we want, then narrow down to max 2-3, make decision matrix perhaps depending on what's narrowed down
 - the AI / non model cars can crash?
 - models figured out by next meeting - claude to narrow down from large list of DRL models, then we can do our own decision matrix for the remaining two-three models to verify claude's results
 - recordings will be our main output, can either create them conditionally (if we did really good)

goals for this meeting:
 - how long does it take to train 
   - we don't know yet, yet to find out
 - we should have a better defined MVP 
   - model merges into highway
   - travel across lanes - how many - 3, fourth for merging that goes away
   - 1 km between two exits, maybe 2km
   - the two exits should be on two different sides (merge on right, then take left exit)
   - react to different personalities - trained on different personalities
   - range - 5/8 cars at once - depending on type of traffic
   - exit without crashing
   - fast as possible
   - metrics? - not getting hit
   - small punishment for switching lanes - no swerving - or maybe make it dynamic and punish lanes at a time
   - randomized three scenarios - basic + default behavior, crowded / rush hour, aggressive cars but not busy
     - train on basic, test on extra
     - validate should be a different track
     - might not be relevant if everything's randomized
   - randomize personality - normal, slow, fast 
      - aggressive/reckless, randomly brakes, intoxicated - extra / after MVP
   - randomize number of cars - between normal and busy
   - randomized tracks - after MVP, number of exits, cars taking exits, etc. etc.
 - what factors to use in the decision matrix
   - *time to train / hardware requirements
   - effectiveness
   - best time
   - we can ask AI - other metrics that don't apply to our situation
   - simplicity / how easy to work with it
   - *how relevant it is for highway-env / our environment
 - TDL for next 
   - OVERALL - draft MVP by 3 weeks (11/10), and the last 5 weeks is debugging and testing
   - readme for a source of truth
     - i can do readme
   - modeling group
     - ask ai for factors for decision matrix beyond time to train and relevance
     - combined that list of three repos, implementations are something we can look at later, and the three of us can do all models
     - each modeling person can independently do the big decision matrix from AI
     - modeling people narrow it down to 5
   - env group
     - get environment set up for MVP such that we can just plug in a model
   - all 5 of us takes one model and researches it then we share next meeting, then narrow it down to 3 from there
   - next week is training a model?
 - when next meeting
   - this time next week? not for jace
   - monday 2-4 next week 10/26





meeting 3 10/9/26 Friday

 - personality for each car
   - personality would have to be something we add onto it. we design it
   - snapshots need to be frequent enough - higher fps
 - change lanes after some time
 - penalize a lot if it hits car
 - different vehicle types - possible something we do later (e.g. trucks)
 - use documentation models as base, tweak them with additional 
 - define MVP, finish 1-2/2-3 weeks before the end, we add one more stuff until we get to that 1-3 week buffer
 - last week 11/30-12/5, 7 weeks including next week
 - MVP by mid november
 - risky/nonrisky drivers
 - training doesn't have to be 100% merging
 - MVP - model that can merge in the highway, travel a certain distance, and change line - 80%, and merge out
 - additional things if we get MVP done: adding personality to driver and other surrounding, comparing models performances, vehicle types
 - 11/15 is 5 weeks and some from today
 - action items: set up simulation / explore library, add models, 
 - Abhi and jace: figure out library
 - connor: list and analyze models
 - aarohi: models
 - Eli: do whatever he wants
 - split into teams by library/models, meet within teams then block 2-3 hours for us to work on this
 - eli can do research too
 - library team: focus on understanding how the library works, getting some visuals up, and a design of the highway
 - model: how a model works, how it interfaces with the library 
 - next monday 10/19 next meeting, 300pm
