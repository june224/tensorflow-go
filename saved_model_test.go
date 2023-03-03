/*
Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package tensorflow

import (
	"os"
	"testing"
	"time"
)

func TestSavedModel(t *testing.T) {
	tags := []string{"serve"}
	bundle, err := LoadSavedModel("../cc/saved_model/testdata/half_plus_two/00000123", tags, nil)
	if err != nil {
		t.Fatalf("LoadSavedModel(): %v", err)
	}
	if op := bundle.Graph.Operation("y"); op == nil {
		t.Fatalf("\"y\" not found in graph")
	}
	t.Logf("SavedModel: %+v", bundle)
	// TODO(jhseu): half_plus_two has a tf.Example proto dependency to run. Add a
	// more thorough test when the generated protobufs are available.
}

func TestSavedModelWithEmptyTags(t *testing.T) {
	tags := []string{}
	_, err := LoadSavedModel("../cc/saved_model/testdata/half_plus_two/00000123", tags, nil)
	if err == nil {
		t.Fatalf("LoadSavedModel() should return an error if tags are empty")
	}
}

func TestLoadModel(t *testing.T) {
	//os.Setenv("LD_PRELOAD","/usr/lib/x86_64-linux-gnu/libjemalloc.so.2")
	//os.Setenv("M_MMAP_MAX","0")
	os.Setenv("M_MMAP_THRESHOLD","1024")
	os.Setenv("M_TRIM_THRESHOLD","1024")
	for i := 0; i < 100000; i++ {
		_, err := LoadSavedModel("/mnt/d/models/ugc_list_nn_v7/1659524286", []string{"serve"}, nil)
		if err != nil {
			panic(err)
		}
		//runtime.GC()
		//if i%10 == 0 {
		//	time.Sleep(10 * time.Second)
		//}
		time.Sleep(200 * time.Millisecond)
	}
	time.Sleep(30 * time.Second)
}
