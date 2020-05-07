import React, { Component } from 'react';

export default class Index extends Component {
    constructor(props) {
        super(props);
        this.state = {
        	ingredients: '',
            recipe: '',
            url: ''
        }
        {this.getURL()}
    }

    getURL = () => {
        //alert("The URL of this page is: " + window.location.href);
        console.log(window.location.href);
        const item = window.location.href.split('=')[1]; 
        const getRecipeUrl = "http://localhost:3001/recipe?item=";
        this.state.url = item;

		fetch(getRecipeUrl + item)
			.then(response => response.json())
			.then(data => this.setState({ingredients: data.ingredients})
			);
        
		
		//console.log(this.state.recipe);

    }

    render() {
        var all_ingredients='';
		var ingr = this.state.ingredients.split(',');
		if(this.state.ingredients){
			var all_ingredients = ingr.map((rec) =>
			<li>{rec}</li>
			);
		}

        return (
            <div>
                <p>Recipe:
                    {this.state.recipe}
                    {all_ingredients}
                </p>
            </div>
        )
    }
}
