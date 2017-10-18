* For 3D face animation, the basic usage is

	`./main_3Dface_evp.out [OPTIONS]`

	Type the following in terminal to get more information:
	
	`./main_3Dface_evp.out -h` or `./main_3Dface_evp.out --help`
	
	Example Usage: Type the following in terminal
	
	`./main_3Dface_evp.out -f data/obj/CYHo.obj -t 1`
	
	There are some prepared obj data files in the `data/obj` directory.

For 3D face animation,  you will see output like

```
dos2unix: converting file data/obj/CYHo.obj to Unix format ...
Loads from "data/obj/CYHo.obj" with color.
"data/obj/CYHo.obj" contains 61961 vertices and 123132 faces.

Verifying boundary ..................... Done.  Elapsed time is 0.131394 seconds.
Reordering vertices .................... Done.  Elapsed time is 0.00719786 seconds.
Constructing Laplacian ................. Done.  Elapsed time is 0.0773089 seconds.
Mapping Boundary ....................... Done.  Elapsed time is 9.60827e-05 seconds.
Solving Eigenvalue Problem ....................... Done.  Elapsed time is 86.1831 seconds.

n = 61173
nnz = 425981
The estimated eigenvalue near 1.1 = 0.0000000000000
```