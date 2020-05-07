import React from 'react';
import ReactDOM from 'react-dom';
import App from './App';
import Ingredients from './Components/fetchImages';
import { BrowserRouter, Route, Switch } from 'react-router-dom';

//ReactDOM.render(<App />, document.getElementById('root'));

ReactDOM.render(
    <BrowserRouter>
      <Switch>
      <Route exact path="/" component={App}/>
      <Route path="/recipe" component={Ingredients}/>
    </Switch>
    </BrowserRouter>,
    document.getElementById('root')
  );


