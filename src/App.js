import React, { Component } from 'react';
import axios from 'axios';
//import 'bootstrap/dist/css/bootstrap.min.css';
import { BrowserRouter as Router, Switch, Route, Link } from 'react-router-dom';
import Ingredients from './Components/fetchImages';

//import logo from './images/uploads/cake.jpg';
//import { BrowserRouter as Router, Switch, Route, Link } from 'react-router-dom';

const BASE_URL = 'http://localhost:5000/';
//const express = require('express');
//const cors = require('cors');
//const app = express();
//app.use(cors());
class App extends Component {
    constructor(props) {
        super(props);
        this.state = {
        	images: [],
        	imageUrls: [],
			message: '',
			similar_images: '',
			classification: '',
			recipe: '',
			ingredients: ''
		}
		this.handleChange = this.handleChange.bind(this);

    }

    selectFiles = (event) => {
    	let images = [];
    	for (var i = 0; i < event.target.files.length; i++) {
            images[i] = event.target.files[i];
        }
        //this.setState({ images, message })
        images = images.filter(image => image.name.match(/\.(jpg|jpeg|png|gif)$/))
        let message = `${images.length} valid image(s) selected`
        this.setState({ images, message })
    }

    uploadImages = () => {
    
    	const uploaders = this.state.images.map(image => {
		    const data = new FormData();
		    data.append("image", image, image.name);
		    
			// Make an AJAX upload request using Axios
			//console.log(data);
	    	return axios.post(BASE_URL + 'upload', data)
	    	.then(response => {
				console.log(response.data);
				this.setState({imageUrl: response.data.imageUrl});
			})
		});
		//const proxyurl = "https://cors-anywhere.herokuapp.com/";
		const classificationUrl = "http://localhost:3001/classification?image_path=public/images/&image_name=";
		const similarityUrl = "http://localhost:3001/similarity?image_path=public/images/";
		/*
		const url2 = "https://hn.algolia.com/api/v1/search?query=redux";
		fetch(url2)
			.then(response => response.json())
			.then(data => this.setState({ classification: data.hits }));
		*/

		//console.log(this.state.imageUrl);
		//fetch(classificationUrl + this.state.imageUrl)
		fetch(classificationUrl)
			.then(response => response.json())
			.then(data => this.setState({ classification: data.predicted })
			);

		axios.all(uploaders).then(() => {
			console.log('done');
		})
		.catch(err => alert(err.message)
		);

	}
	
    similarImages = () => {    
		const classificationUrl = "http://localhost:3001/classification?image_path=public/images/&image_name=";
		const similarityUrl = "http://localhost:3001/similarity?image_path=public/images/&classes=";
		console.log(this.state.classification);
		fetch(similarityUrl + this.state.classification)
			.then(response => response.json())
			.then(data => this.setState({ similar_images: data.similarImages})
			);
		
		console.log(this.state.similar_images);
		

    }
	getIngredients = () => {
		const getRecipeUrl = "http://localhost:3001/imagetorecipe?image_path=public/images/&classes=";
		console.log(this.state.classification);
		fetch(getRecipeUrl + this.state.classification)
			.then(response => response.json())
			.then(data => this.setState({recipe: data.recipe})
			);
		
		console.log(this.state.recipe);
	}

	handleChange(event) {
		this.setState({ingredients: event.target.value});
	}

	cookingSuggestions = (event) =>{
		//console.log('Logging input');
		console.log(this.state.ingredients);
		const ingredientstoimage = "http://localhost:3001/ingredientstoimage?ingredients=";
		console.log(this.state.classification);
		fetch(ingredientstoimage + this.state.ingredients)
			.then(response => response.json())
			.then(data => this.setState({suggestionImages: data.images})
			);

	}
	
	render() {
		//let names = ['name1', 'name2', 'name3'];
		//const images = [];
		//const numbers = [1, 2, 3, 4, 5];
		var all_recipes='';
		var recipes = this.state.recipe.split(',');
		if(this.state.recipe){
			var all_recipes = recipes.map((rec) =>
			<li>{rec}</li>
			);
		}
		var images='';
		if(this.state.similar_images){
			const names = this.state.similar_images.split(',');
			var images = names.map((name) =>
				<a href={"http://localhost:3000/recipe?item="+name}>
					<img src={"food-101/images/"+name} height="100" width="100" style={{marginLeft: '10px'}} />
				</a>
			);
		}
	
		var suggestion_images='';
		if(this.state.suggestionImages){
			const names = this.state.suggestionImages.split(',');
			var suggestion_images = names.map((name) =>
				<a href={"http://localhost:3000/recipe?item="+name}>
					<img src={"data/"+name} height="100" width="100" style={{marginLeft: '10px'}} />
				</a>
			);
		}

		return (
			<Router>				
        	<div>
	        	<br/>
	        	<div className="col-sm-12">
        			<h3>NeuralCook</h3><hr/>
	        		<div className="col-sm-4">
		        		<input className="form-control " type="file" onChange={this.selectFiles} multiple/>
						<br/>
						<div className="col-sm-4">
		            	<button className="btn btn-primary" value="Submit" onClick={this.uploadImages}>Upload</button>
		        		</div><br/><br/><br/>
						<h5> Get recipe and cooking suggestions from ingredients</h5>
		        		<input className="form-control " placeholder="Enter ingredients" type="text" value={this.state.ingredients}  onChange={this.handleChange}/>
						<br/>
		            	<button className="btn btn-primary"   onClick={this.cookingSuggestions}>Fetch suggestions</button>
						<br/><br/>

		        	</div>
					<div className="row col-lg-12" >
							{suggestion_images}
						</div>
		        	{ this.state.message? <p className="text-info">{this.state.message}</p>: ''}
		        	<br/><br/><br/><br/>
	            </div>
	            <br/><br/><br/><br/><br/>
	            <div className="row col-lg-12">
		        	{ 
			          	this.state.imageUrls.map((url, i) => (
				          		<div className="col-lg-2" key={i}>
				          			<img src={BASE_URL + url} className="img-rounded img-responsive" alt="not available"/><br/>
				          		</div>
				          	))
			        }
					<div className="row col-lg-12">
						<img src={this.state.imageUrl} height="100" width="100" style={{marginLeft: '20px'}}/>
						<p style={{color:"blue"}}  style={{fontSize:"18px"}} style={{marginLeft: '20px'}}> {this.state.classification} </p>	
						<button className="btn btn-primary" value="Ingredients" onClick={this.getIngredients} style={{marginLeft: '15px'}}>Get Ingredients</button>
						<ul>{all_recipes}</ul>
	
						<button className="btn btn-primary" value="Similarity" onClick={this.similarImages} style={{marginLeft: '15px'}}>Run similar images</button>
						<hr/>
					</div>

					<div className="row col-lg-12" >
						{images}
					</div>

					
		    	</div>
			</div>

			</Router>	
        );
    }
}

/*
						<img src={"food-101/images/"+this.state.similar_images.split(',')[0]} height="100" width="100" style={{marginLeft: '10px'}} />
						<img src={"food-101/images/"+this.state.similar_images.split(',')[1]} height="100" width="100" style={{marginLeft: '10px'}}/>
						<img src={"food-101/images/"+this.state.similar_images.split(',')[2]} height="100" width="100" style={{marginLeft: '10px'}} />
						<img src={"food-101/images/"+this.state.similar_images.split(',')[3]} height="100" width="100" style={{marginLeft: '10px'}} />
						<img src={"food-101/images/"+this.state.similar_images.split(',')[4]} height="100" width="100" style={{marginLeft: '10px'}} />
						<img src={"food-101/images/"+this.state.similar_images.split(',')[5]} height="100" width="100" style={{marginLeft: '10px'}} />
						<img src={"food-101/images/"+this.state.similar_images.split(',')[6]} height="100" width="100" style={{marginLeft: '10px'}} />
						<img src={"food-101/images/"+this.state.similar_images.split(',')[7]} height="100" width="100" style={{marginLeft: '10px'}} />
						<img src={"food-101/images/"+this.state.similar_images.split(',')[8]} height="100" width="100" style={{marginLeft: '10px'}} />

*/

//					<img src={'images/uploads/1587074014137-cake2.jpg'} height="100" width="100" />

export default App; 			