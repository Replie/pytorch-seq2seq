function set_checkpoint(value) {
    let checkpoint_val = $("#checkpoint_val");
    checkpoint_val.val(value);
    checkpoint_val.text(value);
    $.getJSON("_get_epochs", {
        date: value,
        format: "json"
    })
        .done(function (data) {
            let results = data.epoches;
            let res_element = $("#epochs_list");
            res_element.empty();
            let epochs_val = $("#epochs_val");
            epochs_val.val('');
            epochs_val.text('Epochs');
            results.forEach(function (element) {
                console.log(res_element);
                res_element.append(
                    '<a class="dropdown-item" href="#" onclick="get_epoches($(this))">' + element + '</a>'
                );
            });
        });

}

function get_epoches(param) {
    let checkpoint_val = $("#checkpoint_val").val();
    let value = param.text();
    console.log('checkpoint_val' + checkpoint_val);
    let epochs_val = $("#epochs_val");
    epochs_val.val(value);
    epochs_val.text(value);
    let steps_val = $("#steps_val");
    steps_val.val('');
    steps_val.text('Steps');
    $.getJSON("_get_steps", {
        date: checkpoint_val,
        epoch: value,
        format: "json"
    })
        .done(function (data) {
            let results = data.steps;
            let res_element = $("#steps_list");
            res_element.empty();
            results.forEach(function (element) {
                console.log(res_element);
                res_element.append(
                    '<a class="dropdown-item" href="#" onclick="get_steps($(this))">' + element + '</a>'
                );
            });
        });
}

function get_steps(param) {
    let steps_val = $("#steps_val");
    let value = param.text();
    steps_val.val(value);
    steps_val.text(value);
    $("#submit").removeAttr("disabled");
}

function submitdata() {
    let res_element = $("#results");
    res_element.empty();
    res_element.attr("hidden");
    $.getJSON("_predict", {
        seq_str: btoa($("#input_seq").val()),
        date: $("#checkpoint_val").val(),
        epoch: $("#epochs_val").val(),
        step: $("#steps_val").val(),
        format: "json"
    })
        .done(function (data) {
            let results = data.data.results;
            let res_element = $("#results");
            res_element.removeAttr("hidden");
            res_element.empty();
            results.forEach(function (element) {
                console.log(element);
                res_element.append(
                    "<button type=\"button\" class=\"list-group-item list-group-item-action\">" + element + "</button>"
                );
            });
        });
}