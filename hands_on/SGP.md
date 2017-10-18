* For graph laplacian, the usage is

	`./sgp_main.out [OPTIONS]`
	
	The following are current possible options:
	
	```
	-h,       --help          Display this information
  -f<file>, --file <file>    The graph data file
  -t<num>,  --type <num>     0: simple graph(default if the graph data has 2 columns),
                              1: directed (multi) graph (not supported yet)
                              2: directed weighted graph (not supported yet)
                              3: undirected weighted graph (default if the graph data has 3 columns)
  -p<file>, --para <file>    The parameter setting file
  -e<num>,  --evp <num>      0: None(default), 1: Host, 2: Device
  -l<num>,  --ls <num>       0: None, 1: Direct Host, 2: Direct Device(default), 3: Iterative
  ```
	
	Currently, 2 types of graph are supported:
	* `0`: __simple graph__
	* `3`: __undirected weighted graph__

	Example Usage: Type the following commands in the terminal
	
	* For __simple graph__ , type
		1. Solve eigenvalue problem on device
	
			`./sgp_main.out -f data/graph/ChicagoRegional -e 2`
			
			or
			
			`./sgp_main.out data/graph/ChicagoRegional -t 0 -e 2`
			
		2. Solve linear system with direct device solver

			`./sgp_main.out -f data/graph/ChicagoRegional`
			
		3. Solve linear system with iterative solver
		
			`./sgp_main.out -f data/graph/ChicagoRegional -l 3`
		
	* For __undirected weighted graph__, type
	
		1. Solve eigenvalue problem on device
	
			`./sgp_main.out -f data/graph/moreno_kangaroo_kangaroo -e 2`
			
			or
			
			`./sgp_main.out data/graph/moreno_kangaroo_kangaroo -t 3 -e 2`
			
		2. Solve linear system with direct device solver

			`./sgp_main.out -f data/graph/moreno_kangaroo_kangaroo`
			
		3. Solve linear system with iterative solver
		
			`./sgp_main.out -f data/graph/moreno_kangaroo_kangaroo -l 3`

	
	There are some prepared graph data files in the `data/graph` directory. 

	(__UPDATE__)
	
	You can use the downloaded graph data file as input without modifying the format of the file.