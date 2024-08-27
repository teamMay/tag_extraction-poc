# Tag_extraction-poc

## The goals of this repo

To have a more precise classification than the current “super classes” such as “health”, “food”, and so on. To achieve this, we'll be looking at a number of different methods: more technical word extraction (tf-idf, keyBert etc.) and unsupervised clustering to bring out precise, coherent clusters (“small classes”).

## Organization of the repo

### Tag_extraction_ipde

- The fit_model folder is used to find the right hyperparameters for different models, in order to bring out smaller categories that would give us more information on the question at hand. The algorithms used included hdbscan and kmeans, and a special procedure was adopted, as detailed below.

- the actions_on_cluster folder groups together the various operations performed on the optimal clusters found via the previous fit. This includes, for example, the extraction of keywords for each cluster.

- The dico_num_tag_name_tag folder contains the dicos linking the cluster and the associated tags. These dicos were formed by using the keywords and by a detailed analysis of the questions by hands.

- the dim_reduction folder contains the reducer umap and the code to generate a reducer (deterministically), which is then used to reduce the size of the questions for better clustering and faster execution.

- the visualisation folder contains the code to generate the map of vectors with plotly in order to see the clustering. it also allows to see the questions and the associated tag by passing the mouse over them (if the final dictionary and labels are found).

- the keywords folder contains all the keyword found by the various extraction methods.

- the first_action folder contains the code to import and compute the embeddings of the messages.

- the data folder contains all the data usefull for this task.

### Tag_extraction_midwife

Same as the ipde one

## Results

We end up with hundreds of subclasses for ipde and midwife, based on the first messages sent by the parent. To assign a new message, this is done via a KNN and the map of initial messages, and therefore a search for nearest neighbors.

## Procedure followed

### Initially

An approach using unsupervised metrics was considered (the silhouette score, the Davies-Bouldin Index, the Dunn Index (which takes too long to calculate) or the Calinski-Harabasz Index (or Variance Ratio Criterion)). These metrics were abandoned as not suitable for our data. In fact, they gave results that optimized well-rounded, pretty clusters, except that the distribution of messages isn't like that, and clusters are sometimes shaped like lines, convex set, etc. Clustering was very poor at the time (which is why it wasn't used for midwifes).
We therefore opted for a visualization approach, with a human operator estimating the coherence of the cluster “by hand” (and giving much better results).
Inference (and/or testing) is done by viewing and giving the dictionary (to see the question tag)

### The final approach

First, we import and generate sentence embeddings using a pre-trained model (see the files first_actions and create_embeddings).

Next is the dimensionality reduction phase:

If there is a large amount of data, dimensionality reduction (to vectors of 100 or 50 dimensions) can greatly facilitate the search for optimal clusters (in order to run clustering algorithms faster with over 200,000 data points).
In all cases, we reduce the data to 2D vectors for visualization purposes.
To reduce dimensions, we use the UMAP algorithm (similar to T-SNE but with stochastic rather than complete dimensionality reduction, making it much faster and very useful when handling large datasets). We fit the reducer and save it (which is helpful if new data arrives, as the reducer adjusts to the initial fit data, allowing us to avoid using fit_transform directly by splitting it into two steps). Note: To use UMAP, install umap-learn (and possibly resolve dependency issues).

Here is where the tweaking begins: We perform an initial clustering, either with HDBSCAN or KMeans, depending on which method yields the best results, and we display the 2D vector map each time using Plotly. We then adjust the clustering hyperparameters until we get a satisfactory result (if using HDBSCAN, it’s fine if some data remains unclassified (label -1), as we’ll reclassify it later based on the nearest neighbors with a KNN).

Next, we review each cluster one by one. If a cluster isn’t satisfactory, we separate it again (using only the vectors from that cluster as the dataset) with KMeans, often, though we could also use keyword search, HDBSCAN, etc. This creates new clusters “manually,” and we repeat this process until the information is well-separated.

We then obtain a satisfactory map with tables: the coordinates of the vectors and their associated labels. When new data arrives, we can classify it based on this reference (provided we have a sufficiently representative initial dataset) using a KNN (which is generally more effective than a random forest in this case).
