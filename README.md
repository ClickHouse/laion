# Vector Search - Part 2

This blog post continues our series on Vector Search, building on the previous post where we provided an overview of what Vector search is, its relation to historical inverted index-based approaches, possible use cases for which it currently delivers value and finally some high-level implementation approaches. In this post, we explore Vector Search with relation to ClickHouse in detail through practical examples as well as answering the question "When should i use ClickHouse for Vector Search?"

For our examples, we utilize a ClickHouse Cloud cluster with a total of N cores and XGB of RAM. This examples should, however, be reproducible on an equivalently sized self-managed cluster. Alternatively, start your ClickHouse Cloud cluster today, recieve $300 of credit,  let us worry about the infrastructure and get querying!

## When should i use ClickHouse for Vector Search?

Prior to showing practical examples, lets address the main question most users have when discovering vector support in ClickHouse - When should i use ClickHouse for Vector Search?

Firstly, ClickHouse is not "just" a vector database - its a real-time OLAP database with full SQL support and a wide range of analytical functions to assist users write analytical queries. Some of these functions, and data structures, allow ClickHouse to perform distance operations between vectors - thus allowing it to be used as a vector database. A fully parallelized query pipeline, that utililes the full compute capability of a machine, allows this matching to be performed very quickly, especially when performing exact matching through a linear scan over all rows - delivering performance comparable to dedicated vector databases. High levels of compressions, tunable through custom compression codecs, allow very large datasets to be potentially stored and queried. With a fully-fledged storage engine, ClickHouse is not memory-bound - allowing multi-TB datasets containining embeddings to be queried. Importantly, the capabilities which allow the distance between 2 vectors to be computed are just another SQL function - allowing them to combined with more traidtional SQL fitlering and aggregation capabilties. This allows vectors to be stored and queried alongside metadata (and even rich text).Finally, experimental Approximate Nearest Neighbour (ANN) indices, a feature of vector databases to allo faster approximate matching of vectors, provide a promising development which may further enhance the vector matching capabilities of ClickHouse. 

In summary, ClickHouse is approprate for vector matching when any of the following are true:

- You need to perform linear distance matching over very large vector datasets, and wish to parallelise and distribute this work across many CPU cores with no additional work or configuration.
- You need to match on vector datasets which of a size that relying on memory only indices is not viable either due to cost or avaibility of hardware
- You would benefit from full SQL support when querying your vectors and wish to combine matching will filtering on metadata (and text) and/or aggregation or join capabilities.
- You have related data in ClickHouse already and do not wish to incur the overhead and cost of learning another tool for a few million vectors.
- You have an existing embedding generation pipeline which produces your vectors for storage and do not need this capabiltiy to be native to your storage engine.
- You principally need fast parallelized exact matching of your vectors and do not need a production implementation of ANN (yet!)
- You're an experienced or curious ClickHouse user and trust us to improve our vector matching capabilities and wish to be part of this journey!

While this covers a wide range of use cases, there are cases ClickHouse would be less appropriate as a vector storage engine and you may wish to consider alternatives such as faiss or a dedicated vector database. In the interests of honesty and transparency, ClickHouse is probably not the best choice if:

- Your vector dataset is small and easily fits in memory.
- You have no additional metadata with the vectors and need distance matching and sorting only.
- You have very high QPS, greater than several thousand per second. Typically, for these usecases the dataset will fit in memory and matching times of a few ms are required. While ClickHouse can serve these usecases, a simple in memory index is probably sufficient.
- You need a solution which includes embedding generation capabilties, where a model can be integrated at insert and query time. Vector databases, such as weviate, have pipelines specifically designed for this use case and be more appropriate should you be looking for a solution with OOTB models and accompanying tooling.

With the above in mind, lets explore the vector capablities of ClickHouse.

## Selecting a dataset

As discussed in our previous post, Vector search requires embeddings (vectors representing a contextual meaning) to be generated for a dataset. This requires a model to be produced through a Machine Learning training process, for which the dataset can subsequently be passed to generate an embedding for each object e.g. an image or piece of text. This process is typically involved and the subject of signfiicant research. With some of the latest approaches utilize a class of algorithms known as Transformers, this process is beyond the scope of this specific post. We defer this to a later post, utilizing a pre-prepared embeddings for the focus of this post: Search in ClickHouse. 

Fortunately, the embeddings for test datasets are available for download. Wanting to ensure we utilzied a test set of sufficient size with respect to both number of vectors and their dimensionality, we have selected the LAION 5billion test set. This consists of embeddings, with a dimension of 768, for several bllion public images on the internet and their captions. As of the time writing, we believe this to be largest available dataset of pre-computed embeddings available for testing. Although we utilize a subset of this data for our examples here, later posts will explore the full data in detail. 

As well as providing billions of embeddings with high dimensionality this test set includes metadata useful for illustrating the analytics capabilities of ClickHouse and how they can be used in conjunction with Vector search. This metadata is distributed seperately from the embeddings themselves, which are provided in `.xxx` format. We have combined these to produce a 5TB parquet dataset our users can download and use to reproduce examples.

### The LAION dataset

The LAION dataset was created with the explicit purpose of testing Vector search at scale. With over 5 billion emeddings in total, the dataset was generated for a set of images collected through a public crawl of the internet. An embedding is generated for each image and its associated caption - giving us two embeddings for each object. For this post we have focused on the english subset only, which consists of a reduced 2.2 billion objects. Although each of these objects has 2 embeddings, one for its image and caption repsetiveedly, we represent each object as a single row - giving us almost 2.2 billion rows in total. For each row we include the metadata as columns, which captures information such as the image dimensions and the simularity of the image and caption embedding. This similarity, a cosine distance, allows us to identify objects where the caption and image do not conceptually align - potentially filtering these out in queries (see later).

We would like to acknowledge the effort required by the original authors at XXX to collate this dataset, train the associated model and produce the embeddings for public use. We recommend reading the full process for generating this dataset, which overcame a number of challenging large data engineering challenges such as downloading and resizing billions of images efficiently and in reasonable time.

### The CLIP Model

Before we describe how the LAION dataset can be downloaded and inserted into ClickHouse, we should how the embeddings for the images and caption are generated. They key outcome of this training process is that the embeddings for the two data types are comparable i.e. if the vector for an image and catpure are close, then they can be considered to conceptually. This ability to compare modals (data types), requires a multimodal  machine learning algorithm. For this, the team at XX utilzied the CLIP model by OpenAI. We briefly describe this below and encourage readers to read the full approach as published by OpenAI.



// algoritmic description




### Combining the data

The LAINON dataset is downloadable from a number of sources. Selecting the English subset, we utilized the version hosted by Hugging Face. This service relies Git Large File Storage (GFS?), for which the user needs to install a client to download files from. Once installed, downloading the data requires a single command. For this ensure you have atleast XTB of disk space available.

```bash
Commands
```

Once downloaded, the users is presented with 3 folders. Two of these contain embeddings in the format XXX (effectively a multi-dimensional array format) for the images and captions. A third directory contains Parquet fiels containing the metadata for each image+caption pair. While Parquet is a [well supported format]() in ClickHouse, XXX is currently not readable - although we have plans [to support]().


// image file directories


To load this data into ClickHouse, we wanted to produce a single row per embedding pair with the metadata for enrichment. This would require a process that merged the respective embeddign and metadata for each object. Considering that vectors in ClickHouse can be represented as an array of Floats, a JSON row produced as a result of this process may look like the following:


// JSON example



Fortunately, although the metadata and embeddings are split across multiple files their naming and ordering are aligned. For example, the 1st row in the files `0000.parquet`, `img_emebedding/0000.XXX` are `text_emebedding/0000.XXX` represent the same object and can be concatenated. This makes parallelizing the merging of this data relatively trivial - we can simply parallelize by file name.

The full code for this process can be found [here](). In summary, this utilizes a worker pool of a configurable size. A process can take a job from this pool containing a file name pattern e.g. `0000`. Each file is in turn processed sequentially by a seperate process, with merged objects produced one row at a time. This data is written to a seperate directory in Parquet format using the same file name. With over 2300 original file patterns, each containing approximately 1m entries, this process has sufficient opportunity to be parallelized - a 48 core machine processes the entire dataset in under 90mins. A few key implementation points for those readers interested in the details:

- LAZY iteraiton parquet - batches
- avoid loading XX files on top heap to keep memory down
- img and text files can be missing (so create 0 entries for these)

The final script can be invoked as follows, once dependencies have been installed with Pip.

```bash

// invoke script


```

The final 2313 Parquet files consume around 5TB of disk space.

/// disk space graphic


We should note that whilst Parquet usually acts as an excellent platform independent storage format for analytics data, as noted by our recent blog post, it is sub-optinal as a vector storage medium. None of the supported encodings are particularly effective at compressing sequences of floating point numbers, result in a large file size. Ideally, the XXX format would be more flexible (alllowing support for metadata) or Parquet would support compression techniques specifically targeted for such floating pointing data. We would welcome advice here or suggestions for alternative formats that we could potentially support in ClickHouse.

## Storing Vectors in ClickHouse

With our Parquet files generated, loading this data into ClickHouse requires a few simple steps. The following shows our proposed table schema:

```sql

schema

```

Note how our embeddings are stored as [`Array(Float32)`]() columns. In addition to some useful columns such as `height` and `width` we have a `exif` column. This column contains metadata we can later use for filtering and aggregation. We have mapped as a `Map(String,String)` for flexibility and schema succientness - the column contains over 100k unique meta labels, making it challenging to declare explictly as either a `Tuple` (this would require every sub column to be declared) and inappropriate for the JSON [experimental json object type](). While the Map column allows to easily store this data, accessing a sub key requires all of the keys to be loaded from the column - potentially slowing down queries. We have therefore extracted 5 properties of interest to the root for later analytics. For users interested in the full list of available meta properties, the following query can be used to identify available Map keys and their frequency:

```sql
SELECT
    arrayJoin(mapKeys(exif)) AS keys,
    count() AS c
FROM laion
GROUP BY keys
ORDER BY c DESC
LIMIT 10
```

Our schema also includes add a `_file` column, denoting the original Parquet file from which this data is generated. This allows us to restart a specific file load, should it fail during insertion into ClickHouse.

For future usage, we loaded this data into a public S3 bucket. To insert this data into ClickHouse, users can execute the following query:

```sql
INSERT INTO laion SELECT * FROM s3('')

```
Note: This is a considerable amount of data to load with this querying take around XX hrs. Users can target specific subsets using glob patterns e.g. `s3()`. This can also be used to build a more robust insertion process, since the above is subject to interruptions such as network connectivity issues. We would recommened users thus batch the loading process. The `_file` column can be used to reconcile any loading issues, by confirming the count in ClickHouse with that in the original Parquet files.

For the examples below, we have loaded the first 100m rows into our ClickHouse instance i.e.

```sql
INSERT INTO laion SELECT * FROM s3('')

```

## Compression performance




## Searching Vectors in ClickHouse

In order to perform a vector search in ClickHouse, we require:

 - **An input vector** representing the concept of interest. In our case, this can either be an encoded image or piece of text. This must be encoded using the same CLIP model as that used to generate our data and have the same number of dimensions - 768.
 - **Distance functions** for comparing the search vectors with those stored in ClickHouse. We can compare our input vector to either the image or caption embedding, Note hat this does not require am embedding from the same data type to be compare e.g. embeddings generated from text can be compared to the column `image_embedding` to find conceptually similar images.


### Distance functions

 ClickHouse supports a wide range of Distance functions. For this post, we focus on two which are commonly used in Vector search:

  - cosineDistance(vector1, vector2) - This provides us a with a Cosine similarity between 2 vectors. More specifically, this measures the cosine of the angle between two vectors i.e. the dot product divided by the length. This produces a value between -1 and 1, where 1 indicates the two embeddings are [proportional](https://en.wikipedia.org/wiki/Proportionality_(mathematics)) and thus conceptually identical. A column name and input embedding can be parsed for vector search.
  - L2Distance(vector1, vector2) - This measures the L2 distance between 2 points. Effectively this si the Eucildean distance between two input vectors i.e. length of the line between the points represented by the vectors. For embeddings, the lower distance the mroe conceptually simlar the source objects.

Both of the functions compute a score which can be used to locate similar embeddings, and thus images and captions which are conceptually aligned. The appropriate metric to used depends if you wish to consider magnitude in your score (see [https://cmry.github.io/notes/euclidean-v-cosine](https://cmry.github.io/notes/euclidean-v-cosine)) and depends on the model itself. Other models might use slight variants of these two. For our pre-trained CLIP model, the L2Distance represents the most appropriate distance function based on the [internal scoring used for the official examples](https://codeandlife.com/2023/01/26/mastering-the-huggingface-clip-model-how-to-extract-embeddings-and-calculate-similarity-for-text-and-images/).

For a full list of available distance, as well as vector normalization, functions see [here](). We would love to hear how you utilize these to search your embeddings!

### Generating an input vector

To genertae an input vector, we need to produce an embedding for either a search image or caption. This requires us to download and invoke the CLIP model. This is easily achieved through a simple python script. The isntructions for installing the dependenecies for this script can be found [here](). We show this script below:

```bash
#!/usr/bin/python3
import argparse
from PIL import Image
import clip
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='generate',
        description='Generate CLIP embeddings for images or text')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--text', required=False)
    group.add_argument('--image', required=False)
    parser.add_argument('--limit', default=1)
    parser.add_argument('--table', default='laion_1m')
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using {device}")
    device = torch.device(device)
    model, preprocess = clip.load("ViT-L/14")
    model.to(device)
    images = []
    if args.text:
        inputs = clip.tokenize(args.text)
        with torch.no_grad():
            print(model.encode_text(inputs)[0].tolist())
    elif args.image:
        image = preprocess(Image.open(args.image)).unsqueeze(0).to(device)
        with torch.no_grad():
            print(model.encode_image(image)[0].tolist())
```


This version of the script accepts either text or an image path as input, outputing the embedding to the command line for copying into a ClickHouse query. Note that this will exploit cuda-enabled GPUs if present. This can make a dramatic difference to the generation time - when tested on a Mac M1 2021 generation without gpu support takes around 1sec vs Nsecs on a [p3.2xlarge](https://aws.amazon.com/ec2/instance-types/p3/) with 1 GPU core.

Below we convert the text "a sleepy [ridgeback dog](https://en.wikipedia.org/wiki/Rhodesian_Ridgeback)" into a embedding.

```
python generate.py --text "a sleepy ridgeback dog"

[0.5736801028251648, 0.2516217529773712, ...,  -0.6825592517852783]
```

For purposes of brevity, we have cropped the full embedding result which can be found [here]().

### Puting it together

Using the embedding generated above and the Eucilean distance function, we can identify conceptually similar images. The query below matches on the `image_embedding` column, computing a score and sorting by this. We require a `similarity` score of atleast 0.2, to ensure the original text and image embeddings are considered similar - thereby filtering out noise.

```
SELECT
    url,
    caption,
    L2Distance(image_embedding, [0.5736801028251648, 0.2516217529773712, ...,  -0.6825592517852783]) AS score
FROM laion_10m
ORDER BY score ASC
LIMIT 2
FORMAT Vertical

Row 1:
──────
url:     https://thumb9.shutterstock.com/image-photo/stock-photo-front-view-of-a-cute-little-young-thoroughbred-african-rhodesian-ridgeback-hound-dog-puppy-lying-in-450w-62136922.jpg
caption: Front view of a cute little young thoroughbred African Rhodesian Ridgeback hound dog puppy lying in the woods outdoors and staring.
score:   12.262669854099084

Row 2:
──────
url:     https://m.psecn.photoshelter.com/img-get2/I0000_1Vigovbi4o/fit=180x180/fill=/g=G0000x325fvoXUls/I0000_1Vigovbi4o.jpg
caption: SHOT 1/1/08 3:15:27 PM - Images of Tanner a three year-old male Vizsla sleeping in the sun on the couch in his home in Denver, Co. The Hungarian Vizsla, is a dog breed originating in Hungary. Vizslas are known as excellent hunting dogs, and also have a level personality making them suited for families. The Vizsla is a medium-sized hunting dog of distinguished appearance and bearing. Robust but rather lightly built, they are lean dogs, have defined muscles, and are similar to a Weimaraner but smaller in size. The breed standard calls for the tail to be docked to two-thirds of its original length in smooth Vizslas and to three-fourths in Wirehaired Vizslas..(Photo by Marc Piscotty/ © 2007)
score:   12.265198669961046

2 rows in set. Elapsed: 2.637 sec. Processed 10.00 million rows, 32.69 GB (3.79 million rows/s., 12.40 GB/s.)

```

These results seem sensible based on the caption. As an alternative we can reverse the modals, passing [an image]() of a sleeping dog (the author's dog, a ridgeback 'Kibo' for you dog lovers) to the encoding function. We then repeat the above query, using the `text_embedding` column. 

![Kibo](/opt/laion/kibo.jpg)


```bash
python generate.py --image ridgeback.jpg

[0.17179889976978302, 0.6171532273292542, ...,  -0.21313616633415222]
```


```sql
SELECT
    url,
    caption,
    L2Distance(text_embedding, [0.17179889976978302, ..., -0.21313616633415222]
) AS score
FROM laion_10m
ORDER BY score ASC
LIMIT 2
FORMAT Vertical


Row 1:
──────
url:     https://i.pinimg.com/236x/ab/85/4c/ab854cca81a3e19ae231c63f57ed6cfe--submissive--year-olds.jpg
caption: Lenny is a 2 to 3 year old male hound cross, about 25 pounds and much too thin. He has either been neglected or on his own for a while. He is very friendly if a little submissive, he ducked his head and tucked his tail a couple of times when I...
score:   17.903361501636233

Row 2:
──────
url:     https://d1n3ar4lqtlydb.cloudfront.net/c/a/4/2246967.jpg
caption: American Pit Bull Terrier/Rhodesian Ridgeback Mix Dog for adoption in San Clemente, California - MARCUS = Quite A Friendly Guy!
score:   17.90681726342255

2 rows in set. Elapsed: 2.628 sec. Processed 10.00 million rows, 32.69 GB (3.80 million rows/s., 12.44 GB/s.)


```

These results again seem sensible. While sufficient for adhoc testing, this can be alittle tedious when testing a large number of images - copying 768 floating point values is awkward!. We have thus provided a simple result generator [search,py](), which encodes the passed image or text and also executes the query, rendering the query results as a local html file. This file is then automatically be opened in the local browser.  The result file for the above query is shown below:

```bash
python search.py --image ridgeback.jpg

```



For both of these above examples, we have mached embeddings for different modals e.g.  embeddings from image inputs are matched against the `text_embedding` column and vise versa. This aligns with the original model training as described earlier, and is the intended application. While matching input embeddings against the same type has been explore, results are typically [mixed](https://github.com/openai/CLIP/issues/1).

### Linear scans and pre-filtering

All of the previous examples rely on passing an input vector and linearly scanning every row for the those with the closest distance.

- filte

### Exploiting data types and Full SQL support




- agg

- inverted indices




## Improving compression

Our previous schema and resulting compression statistics were based on storing our vectors as the type `Array(Float32)`.For some models, 32-bit floating point precision is not required and similar matching quality can be achieved by reducing this to 16 bits. If we are able to store each of the values of our vector with this lower precision, it potentially has the advantage of reducing our total data size and storage requirements. While ClickHouse does not have a native 16 bit floating point type, we can still reduce our precision to 16 bits and reuse the `Float32` type which each value simply padded with zeros.  These zeros will be efficiently compressed with the ZSTD codec (the standard in ClickHouse Cloud) reducing our compressed storage requirements. 

In order to not impact the range of our vector values, and only reduce the precision, we need to ensure we encode our 16 bit floating point values properly. Fortunately, google invented the BFloat16 type for Machine Learning use cases where floating point numbers of a lower precision are tolerable. This scheme simply requires the truncation of last 16 bits of a 32 bit floating point number - assuming the latter is using the IE XXX encoding. This is standard on most modern CPUs and the case with ClickHouse. A BFloat16 type and/or functionj to perform this truncation is [not currently native]() to ClickHouse but can easily be replicated with other functions. We do this below for the `image_embedding` and `text_embedding` columns.  Here we select all rows from the table `laion_100m` (containing 100m rows), inserting them into the table `laion_100m_v2` using an `INSERT INTO SELECT` clause. During the `SELECT` we transform the values in the embeddings to a BFloat16 representation.

```sql
insert into default.laion_100m_v2 
select key, url, caption, similarity, width, height, original_width, original_height, status, NSFW, exif, 
arrayMap(x -> reinterpretAsFloat32(bitAnd(reinterpretAsUInt32(x), 4294901760)), image_embedding) as image_embedding, arrayMap(x -> reinterpretAsFloat32(bitAnd(reinterpretAsUInt32(x), 4294901760)), text_embedding) as text_embedding 

from laion_100m
```

This BFloat16 conversion is achieved using an `arrayMap` function i.e. `arrayMap(x -> reinterpretAsFloat32(bitAnd(reinterpretAsUInt32(x), 4294901760)), image_embedding)`. This iterates over every value `x` in a vector embedding, executing the transformation ` reinterpretAsFloat32(bitAnd(reinterpretAsUInt32(x), 4294901760))` - this interprets the binary sequence as an Int32 using the function `reinterpretAsUInt32` and performs a `bitAnd` with the value `4294901760`. This latter value is the binary sequence `000000000000000001111111111111111`. This operation therefore zeros the trailing 16 bits, performing an effective truncation. The resulting binary value is then re-intepreted as a float32. We illustrate this process below:


//image


As shown below this has had the effect of reducing our compressed data by over X% - 0s compress really well.

//compression stats

We can further reduce our on disk size by increasing our compression level for ZSTD to 3. As shown below, this futher compresses our `*_embedding` columns by around X%.

//compression stats

An obvious question might be how this reduction in precision impacts our ability to represent real word concepts in our vectors and resulting search quality - we have, afterall, reduced the information encoded in our multi-dimension space and effectively condensed our vectors "closer" together. Below we show the results for the earlier "a sleepy ridgeback dog" query using our new `laion_100m_v2` table and our `search.py` script.

```bash
python search.py --table laion_100m_v2 --query "a sleepy ridgeback dog"
```

//image

Whle the results remain relevant, and probably acceptable, there is clearly some reduction in search quality. This was alittle suprising as we expected it to have minmal effect initially. The `Float32` encoding is already a precision reduction on the `Float64` values produced by the model, however. Users will need to test this precision reduction technique on their specific model and dataset, with results likely varying case by case.

## Scaling Vector search

Our previous examples have used a 10m sample of the 2 billion dataset. Due to the huge number of images in the dataset, the actual results produced by our search phrases and images remained acceptable. However, users will still be curious as to how the linear scan techniques perform as the number of images increases.



### Approximate Nearest Neighbour (ANN)


### Approximating ANN



## Vector fun

After reading an [interesting blog post]() on how vector math can be used to move around a high dimensionality space containing concepts, we thought it might interesting to see if the same could be achived with our CLIP generated embeddings. As noted in the referenced post, embeddings have the interesting property that simple math can be used to move between concepts.

For example, suppose we have embeddings for the words `berlin`, `germany`, `portugal` and `bridge`. The following mathematical operation can be performed on their respective vectors.

`(berlin - germany) + (portugal + bridge)`

Knowing that berlin is the captial of Germany, if we logically subtract and add the above concepts, we might deduce the result would represent the bridge in Lisbon - the captial of portugal. Interestingly, the resulting embedding should also close to the same concept in our respective space.

Testing this idea, we enhanced our simple `search.py` script to support a basic parser that could accept input similar to the above. This parser supports the operations `+`, `-`, `*` and `/`, as well `'` to denote multi-term input, and is exposed through a `concept_math` command.
Thanks to the amazing `pyparsing` library, building a parser for this grammar is trivial. In summary, the above phrase would be parsed into the following syntax tree:


//image


We can inturn recursively compute the vectors for the text terms (the leafs) in the above tree. Branches can then combined using the equivalent vector functions in ClickHouse for the specified specified mathematical operator. This process is performed, depth first, resolving the entire tree to a single query (which should represent the equivalent concept). We illustrate this below:


// image


Finally, this function is matched on the `image_embedding` column using the same process as a standard search. The above would therefore resolve to the follow query:


// image



We show the results for this below, matching on the 10m row sample:

```bash
python search.py compute_concept "(berlin - germany) + (portugal + bridge)"
```




Cool! it works! That is indeed the [famous bridge in Lisbon](). For San Francisco based readers thinking that looks like a bridge near them, yes it shares some history with the golden gate!



//image


Finally, we thought enhancing the grammar parser to support integer constants could be useful. Specifically, maybe we could see if the midpoint between 2 contrasting concepts produced something interesting. For example,the concept represented by the mid point between `X` and `Y` might represent `Z`. Mathematically, this can be represented as `(X+Y)/2`.

Executing this search actually produced something interesting:


//image

This is unlikely to make sense for all concept pairs, it shows another interesting possibility for combining vectors.
There are no doubt other cases where this basic vector math can be useful. We'd love to hear about any examples!

### Exploiting UDFs

Upto now we've relied on performing our vector generation outside of ClickHouse, passing the generated embedding at query time from our `search.py` script. While this is sufficient, it would be nice if we avoid having to run Python ourselves and simply pass text or image paths (or even urls!) in the SQL query themselves. For example it might be nice to be able to execute the following:


```sql
SELECT url, caption, L2Distance(generateEmbeddingFromText("a sleepy ridgeback dog")) as score FROM laion_100m ORDER BY score ASC LIMIT 10


SELECT url, caption, L2Distance(generateEmbeddingFromUrl("https://dogpictures.com/ridgeback.jpg")) as score FROM laion_100m ORDER BY score ASC LIMIT 10
```

Note how the above assumes we can generate an embedding from text or an image url, using the functions `generateEmbeddingFromText` and `generateEmbeddingFromUrl` respectively. This might be a more natural way for ClickHouse users to interact with our vectors. To achieve this, we can exploit [User Defined Functions]() (UDFs). ClickHouse can interact with a Python UDF through stdout and stdin. We can adapt our earlier `generate.py` to accept text as input, generate an embedding per usual using our model, before writing the result to stdout for ClickHouse to consume. We show the resulting [`text_udf.py`]() below.

```python
udf

```


This script can then exposed through a custom function `generateEmbeddingFromText` in ClickHouse, as shown below.



With the function registered, we can now utilise this as shown in our previous example:

```sql



```

For our similar `generateEmbeddingFromUrl` function, we add another UDF based on the following python script [`url_udf.py`]().

```python



```

Using this, we can now query using images.


```sql




```

Completing this, we can expose our earlier concept math capabiltiies with a `concept_udf.py` and function `generateEmbeddingFromQuery`. Users can find the full python script [here](). This uses a slightly different approach to our earlier implementation which used ClickHouse functions to perform the vector math. For simplicity, we perform the math inside the python script itself using `numpy` i.e. instead of returning a query, the script outputs a computed query vector to stdout. The result is, however, the same:


```sql



```

Hopefully these examples have provided some inspiraton for combining User defined functions, embedding models and vector search!

## Conclusion

