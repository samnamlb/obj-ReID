# COMP560 Course Project — Object Re-Identification

This repository contains our group’s work for the **COMP560 Object Re-Identification (ReID) course project**.

The goal of the project is to implement a **ReID model** that maps images of objects (animals, vehicles, people, etc.) into discriminative embeddings, enabling retrieval of the same individual across different viewpoints, cameras, and conditions.

---

```
.
|
|---models
|    |---__init__.py
|    |---model.py
|
|---testing
|    |---dummytest.py 
```
`__init__.py` makes models a module to use via `from models.model import StudentModel`.

`model.py` includes **StudentModel** class that: 
  1. Loads a pretrained ResNet-50 model
  2. Projects features into 512-dimensional embedding (does not store it)
  3. Normalizes output

`dummytest.py` is a *temporary* sanity check that tests for:
  - Shape correctness (each embedding is size of 512)
  - All embeddings have length of 1
  - Same image produces exact embedding

### **[!] NOTE:** This is *NOT* the full implementation
-  This is just the base code to make the image detection to work at **bare minimum**
-  No finetuning and/or training has been added
-  Once we get the dataset, *then* we can move on to performance metrics & evaluation + finetuning w/ **Tinker API**
