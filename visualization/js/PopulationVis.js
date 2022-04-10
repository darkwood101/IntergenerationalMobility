
/*
 * PopulationVis - Object constructor function
 * @param _data					
 */

class PopulationVis {


	constructor(_data) {
		this.data = _data;

		this.initVis();
	}


	/*
	 * Initialize visualization (static content, e.g. SVG area or axes)
	 */

	initVis() {
		let vis = this;
		console.log(vis.data)

		vis.margin = { top: 40, right: 0, bottom: 60, left: 60 };

		
		vis.width = window.innerWidth - vis.margin.left - vis.margin.right;
    	vis.height = window.innerHeight - vis.margin.top - vis.margin.bottom;

		// SVG drawing area
		vis.svg = d3.select("#population").append("svg")
			.attr("width", vis.width + vis.margin.left + vis.margin.right)
			.attr("height", vis.height + vis.margin.top + vis.margin.bottom)
			.append("g")
			.attr("transform", "translate(" + vis.margin.left + "," + vis.margin.top + ")");

        vis.x = d3.scaleOrdinal()
            .domain([1, 2, 3])
            .range([50, 200, 340])

    
		console.log(vis.data)
		// (Filter, aggregate, modify data)
		vis.wrangleData(0);
	}



	/*
	 * Data wrangling
	 */

	wrangleData(selected) {
		let vis = this;


		// Update the data
		if(selected == 1){
			vis.data.forEach(function(d){
				let rand = Math.random();
				if(rand < 0.5){
					d.advantage = 1
				}
			})
		}

		// Update the visualization
		vis.updateVis();
	}



	/*
	 * The drawing function - should use the D3 update sequence (enter, update, exit)
	 * Function parameters only needed if different kinds of updates are needed
	 */

	updateVis() {
		let vis = this;
		
		vis.node = vis.svg.append("g")
                .selectAll("circle")
                .data(vis.data)
                .enter()
                .append("circle")
				.merge(vis.svg)
				//.transition()
				.attr("r", 29)
				.attr("cx", vis.width / 2)
				.attr("cy", vis.height / 2)
				.style("fill", function(d){ 
					if(d.advantage == 1){
						return '#EC7063';
					}     
					else{
						return '#a2dbc0';
					}
				})
				// .merge()
				// .transition()
				.style("fill-opacity", 0.85)
				.attr("stroke", "black")
				.style("stroke-width", 4)
				
		vis.simulation = d3.forceSimulation()
		.force("x", d3.forceX().strength(0.5).x(d => vis.x(d.group)))
		.force("y", d3.forceY().strength(0.1).y(vis.height/2 ))
		.force("center", d3.forceCenter().x(vis.width / 2).y(vis.height / 2)) // Attraction to the center of the svg area
		.force("charge", d3.forceManyBody().strength(1)) // Nodes are attracted one each other of value is > 0
		.force("collide", d3.forceCollide().strength(.1).radius(32).iterations(1)) // Force that avoids circle overlapping
				
		// Apply these forces to the nodes and update their positions.
		// Once the force algorithm is happy with positions ('alpha' value is low enough), simulations will stop.
		vis.simulation
			.nodes(vis.data)
			.on("tick", function(d){
				vis.node
					.attr("cx", d => d.x)
					.attr("cy", d => d.y)
			});

		vis.svg.exit().remove();

		

	}
}