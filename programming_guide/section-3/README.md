<!---//===- README.md ---------------------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Section 3 - Data Movement (Object FIFOs)</ins>

In this section of the programming guide, we introduce the Object FIFO high-level abstraction used to describe the data movement within the AIE array. At the end of this guide you will:
1. have a high-level understanding of the communication primitive API,
2. have learned how to initialize and access an Object FIFO through meaningful design examples,
3. understand the design decisions which led to current limitations and/or restrictions in the Object FIFO design,
4. know where to find more in-depth material of the Object FIFO implementation and lower-level lowering.

To understand the need for a data movement abstraction we must first understand the hardware architecture with which we are working. The AIE array is a spatial compute architecture with explicit data movement requirements. Each compute unit of the array works on data that is stored within its L1 memory module and that data needs to be explicitly moved there as part of the AIE's array global data movement configuration. This configuration involves several specialized hardware resources which handle the data movement over the entire array in such a way that data arrives at its destination without loss. The Object FIFO provides users with a way to specify the data movement in a more human comprehensible and accessible manner without sacrificing some of the more advanced control possibilities which the hardware provides.

## Initializing an Object FIFO

An Object FIFO represents the data movement connection between a point A and a point B. In the AIE array, these points are AIE tiles (see [Section 1 - Basic AI Engine building blocks](../section-1/)). Under the hood, the data movement configuration for different types of tiles (Shim tiles, Mem tiles, and compute tile) is different, but there is no difference between them when using an Object FIFO. 

To initialize an Object FIFO, users can use the `object_fifo` class constructor (defined in [aie.py](../../python/dialects/aie.py)):
```
class object_fifo:
    def __init__(
        self,
        name,
        producerTile,
        consumerTiles,
        depth,
        datatype,
        dimensionsToStream=None,
        dimensionsFromStreamPerConsumer=None,
    )
```
We will now go over each of the inputs, what they represents and why they are required by the abstraction. We will first focus on the mandatory inputs and in a later section of the guide on the default valued ones (see Data Layout Transformations [subsection](#data-layout-transformations)).

First of all, an Object FIFO has a unique `name`. It functions as an ordered buffer that has `depth`-many objects of specified `datatype`. Currently, all objects in an Object FIFO have to be of the same datatype. The datatype is a tensor-like attribute where the size of the tensor and the type of the individual elements are specified at the same time (i.e. `<16xi32>`).

An Object FIFO is created between a producer or source tile and a consumer or destination tile. Below, you can see an example of an Object FIFO created between producer tile A and consumer tile B:
```
A = tile(1, 2)
B = tile(1, 3)
of0 = object_fifo("objfifo0", A, B, 1, T.memref(256, T.i32()))
```
The created Object FIFO is stored in the `0f0` variable and is named `objfifo0`. It has a depth of `1` object and its datatype is `<256xi32>`.

As you will see in the Key Object FIFO Connection Patterns [subsection](#key-object-fifo-connection-patterns), an Object FIFO can have multiple consumer tiles, which describes a broadcast connection from the source tile to all of the consumer tiles. As such, the `consumerTiles` input can be either a single tile or an array of tiles. This is not the case for the `producerTile` input as currently the Object FIFO does not support multiple producers.

*Note: When specified as a number, the `depth` of an Object FIFO may be adjusted at compile-time based on the access patterns of its producer and consumers (see Object FIFO Access Patterns [subsection](#object-fifo-access-patterns)).*

## Object FIFO Access Patterns

An Object FIFO can be accessed by the processes running on the producer and consumer tiles registered to it. Before a process can have access to the objects it has to acquire them from the Object FIFO. This is because the Object FIFO is a synchronized communication primitive and two processes may not access the same object at the same time. To acquire objects users should use the acquire function of the `object_fifo` class:
```
def acquire(self, port, num_elem)
```
Based on the `num_elem` input representing the number of acquired elements, the acquire function will either directly return an object, or an array of objects that can be accessed in an array-like fashion.

Once a process has finished working with an object and has no further use for it, it should release it so that another process will be able to acquire and access it. To release one or multiple objects users should use the release function of the `object_fifo` class:
```
def release(self, port, num_elem)
```
A process may release one, some or all of the objects it has acquired. The release function will release objects from oldest to youngest in acquired order. If a process does not release all of the objects it has acquired, then the next time it acquires objects the oldest objects will be those that were not released. This functionality is intended to achieve the behaviour of a sliding window through the Object FIFO primitive. (TODO: add link to ref design or subsection)

The patterns in which a producer or a consumer process acquires and releases objects from an Object FIFO are called `access patterns`. We can specifically refer to the acquire and release patterns as well.

## Key Object FIFO Connection Patterns

#### Reuse

#### Broadcast

* multiple consumers (consumerTiles can be either a Tile or an array of Tiles)
* normal and with skip connection (depth can be either a number or an array)

#### Link

```
class object_fifo_link(ObjectFifoLinkOp):
    def __init__(
        self,
        fifoIns,
        fifoOuts,
    )
```

#### Link & Distribute

#### Link & Join

#### Data Layout Transformations

* dimensionsToStream, dimensionsFromStreamPerConsumer

* Point to more detailed objectfifo material in Tutorial
