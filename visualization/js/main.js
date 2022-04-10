selected = 0
// randomly generate data
all_data = []
for (let i = 0; i < 20; i++){
    let rand = Math.random();
    let entry = {
        advantage: 1,
    }
    if(rand > 0.5){
        entry.advantage = 0
    }
    all_data.push(entry)
}

var population = new PopulationVis(all_data)
//var lineGraph = new LineVis(all_data)

function update(){
    console.log("a")
    selected = 1
    population.wrangleData(selected);
}